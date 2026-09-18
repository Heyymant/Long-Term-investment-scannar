"""XBRL parser for NSE quarterly results filings.

NSE publishes each company's results as an Ind-AS XBRL document. Those files
contain the actual numbers - revenue, PAT, EPS, borrowings - which is what
turns the free NSE feed into a genuine fundamentals source.

Two practical wrinkles handled here:

1. **Namespaces vary** between taxonomy versions, so tags are matched on the
   local name rather than a fully-qualified one.
2. **Multiple contexts per document.** A Q3 filing typically reports the
   quarter, the year-to-date period and the prior-year comparatives all in one
   file. Picking the wrong context silently mixes a 9-month figure into a
   quarterly series, which would wreck any SUE calculation - so contexts are
   parsed and the one matching the filing's own period is selected.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

# Local XBRL tag name -> our canonical field. Several spellings exist across
# taxonomy versions, so each field lists its known aliases.
TAG_MAP: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromOperations",
        "IncomeFromOperations",
        "RevenueFromOperationsNet",
    ),
    "other_income": ("OtherIncome",),
    "total_income": ("Income", "TotalIncome"),
    "total_expenses": ("Expenses", "TotalExpenses"),
    "profit_before_tax": (
        "ProfitBeforeTax",
        "ProfitLossBeforeTax",
        "ProfitBeforeExceptionalItemsAndTax",
    ),
    "tax_expense": ("TaxExpense", "TotalTaxExpense"),
    "net_income": (
        "ProfitLossForPeriod",
        "ProfitLossForThePeriod",
        "NetProfitLossForThePeriod",
    ),
    "eps_basic": (
        "BasicEarningsLossPerShareFromContinuingOperations",
        "BasicEarningsLossPerShare",
        "BasicEarningsPerShare",
    ),
    "eps_diluted": (
        "DilutedEarningsLossPerShareFromContinuingOperations",
        "DilutedEarningsLossPerShare",
    ),
    "equity_capital": ("PaidUpValueOfEquityShareCapital", "EquityShareCapital"),
    "face_value": ("FaceValueOfEquityShareCapital",),
    "depreciation": ("DepreciationDepletionAndAmortisationExpense",),
    "finance_cost": ("FinanceCosts",),
}

# Descriptors that identify the reporting entity/scope of the filing.
SCOPE_TAGS = (
    "NatureOfReportStandaloneConsolidated",
    "DateOfStartOfReportingPeriod",
    "DateOfEndOfReportingPeriod",
)


@dataclass
class XBRLContext:
    """One reporting period declared in the document."""

    context_id: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    instant: pd.Timestamp | None = None
    has_dimensions: bool = False

    @property
    def duration_days(self) -> float:
        if self.start is None or self.end is None:
            return float("nan")
        return float((self.end - self.start).days)

    @property
    def is_quarterly(self) -> bool:
        d = self.duration_days
        return 60 <= d <= 115 if d == d else False


@dataclass
class XBRLFacts:
    """Parsed results for one filing."""

    isin: str | None = None
    symbol: str | None = None
    period_start: pd.Timestamp | None = None
    period_end: pd.Timestamp | None = None
    scope: str | None = None            # Standalone | Consolidated
    values: dict[str, float] = field(default_factory=dict)
    context_used: str | None = None

    def get(self, field_name: str) -> float | None:
        return self.values.get(field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "isin": self.isin,
            "symbol": self.symbol,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "scope": self.scope,
            "context_used": self.context_used,
            **self.values,
        }


def _local(tag: str) -> str:
    """Strip the namespace from an element tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def parse_contexts(root: ET.Element) -> dict[str, XBRLContext]:
    """Read every <context> so we can tell a quarter from a year-to-date."""
    contexts: dict[str, XBRLContext] = {}
    for el in root.iter():
        if _local(el.tag) != "context":
            continue
        ctx_id = el.get("id")
        if not ctx_id:
            continue

        start = end = instant = None
        has_dims = False
        for child in el.iter():
            name = _local(child.tag)
            text = (child.text or "").strip()
            if name == "startDate" and text:
                start = pd.to_datetime(text, errors="coerce")
            elif name == "endDate" and text:
                end = pd.to_datetime(text, errors="coerce")
            elif name == "instant" and text:
                instant = pd.to_datetime(text, errors="coerce")
            elif name in ("explicitMember", "typedMember"):
                # Dimensional contexts are segment breakdowns, not the
                # consolidated totals we want.
                has_dims = True

        contexts[ctx_id] = XBRLContext(ctx_id, start, end, instant, has_dims)
    return contexts


def select_context(
    contexts: dict[str, XBRLContext],
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> str | None:
    """Pick the context matching the filing's own reporting quarter.

    Preference order: an exact period match, then the shortest non-dimensional
    duration that looks like a quarter. Falling back to "first context found"
    is how year-to-date figures leak into quarterly series.
    """
    candidates = [c for c in contexts.values() if not c.has_dimensions and c.start and c.end]
    if not candidates:
        return None

    if period_end is not None:
        target_end = pd.Timestamp(period_end).normalize()
        exact = [
            c for c in candidates
            if c.end is not None and abs((c.end.normalize() - target_end).days) <= 3
        ]
        if exact:
            if period_start is not None:
                target_start = pd.Timestamp(period_start).normalize()
                closer = [
                    c for c in exact
                    if c.start is not None and abs((c.start.normalize() - target_start).days) <= 5
                ]
                if closer:
                    return closer[0].context_id
            # Same end date: the shortest span is the quarter, not the YTD.
            exact.sort(key=lambda c: c.duration_days)
            return exact[0].context_id

    quarterly = [c for c in candidates if c.is_quarterly]
    pool = quarterly or candidates
    pool.sort(key=lambda c: (c.end is None, -(c.end.timestamp() if c.end else 0), c.duration_days))
    return pool[0].context_id if pool else None


def parse_xbrl(
    content: str,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> XBRLFacts | None:
    """Extract canonical financial fields from one XBRL document."""
    if not content or not content.strip():
        return None
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        log.debug("XBRL parse failed: %s", exc)
        return None

    contexts = parse_contexts(root)
    chosen = select_context(contexts, period_start, period_end)

    # Invert the alias map for a single pass over the document.
    lookup: dict[str, str] = {}
    for canonical, aliases in TAG_MAP.items():
        for alias in aliases:
            lookup[alias.lower()] = canonical

    facts = XBRLFacts(period_start=period_start, period_end=period_end, context_used=chosen)
    best_by_field: dict[str, tuple[int, float]] = {}   # field -> (priority, value)

    for el in root.iter():
        name = _local(el.tag)
        text = (el.text or "").strip()
        if not text:
            continue

        lowered = name.lower()
        if lowered in ("isin", "isincode", "symbol", "nseSymbol".lower(), "scripcode"):
            if "isin" in lowered and not facts.isin:
                facts.isin = text
            elif "symbol" in lowered and not facts.symbol:
                facts.symbol = text
            continue
        if name in SCOPE_TAGS and name.startswith("NatureOfReport"):
            facts.scope = text
            continue

        canonical = lookup.get(lowered)
        if canonical is None:
            continue

        value = _to_float(text)
        if value is None:
            continue

        ctx_ref = el.get("contextRef")
        # Prefer the chosen context; accept a non-dimensional one otherwise.
        if chosen is not None and ctx_ref == chosen:
            priority = 0
        elif ctx_ref in contexts and not contexts[ctx_ref].has_dimensions:
            priority = 1
        elif ctx_ref is None:
            priority = 2
        else:
            continue

        existing = best_by_field.get(canonical)
        if existing is None or priority < existing[0]:
            best_by_field[canonical] = (priority, value)

    facts.values = {k: v for k, (_, v) in best_by_field.items()}
    if not facts.values:
        return None

    if facts.period_start is None and chosen and contexts.get(chosen):
        facts.period_start = contexts[chosen].start
    if facts.period_end is None and chosen and contexts.get(chosen):
        facts.period_end = contexts[chosen].end

    return facts


def _to_float(text: str) -> float | None:
    cleaned = re.sub(r"[,\s]", "", text)
    if not cleaned or cleaned in ("-", "NA", "NIL"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def facts_to_frame(facts_list: list[XBRLFacts]) -> pd.DataFrame:
    """Collect parsed filings into a tidy frame."""
    rows = [f.to_dict() for f in facts_list if f is not None]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("period_start", "period_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
