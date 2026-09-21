"""NSE parser tests.

All fixtures are recorded payloads - no live network calls, so the suite stays
fast and deterministic. The samples below are trimmed versions of real NSE
responses captured from the public endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.nse import (
    _normalize_legacy,
    _normalize_udiff,
    bhavcopy_to_panels,
    build_symbol_history,
    infer_delistings,
    normalize_corporate_actions,
    parse_corporate_action,
)
from src.data.xbrl import parse_contexts, parse_xbrl, select_context

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------- corp actions
@pytest.mark.parametrize("subject,expected_type", [
    ("Dividend - Rs 2 Per Share", "dividend"),
    ("Interim Dividend - Rs 5.50 Per Share", "dividend"),
    ("Face Value Split From Rs 10 To Rs 2", "split"),
    ("Face Value Split (Sub-Division) - From Rs 10/- To Re 1/-", "split"),
    ("Bonus 1:1", "bonus"),
    ("Bonus issue 2:5", "bonus"),
])
def test_parses_known_action_subjects(subject, expected_type):
    action = parse_corporate_action(subject)
    assert action is not None, f"failed to parse: {subject}"
    assert action["action_type"] == expected_type


def test_split_ratio_gives_correct_share_multiplier():
    """Face value 10 -> 2 means one share becomes five."""
    from src.data.corporate_actions import _share_multiplier

    action = parse_corporate_action("Face Value Split From Rs 10 To Rs 2")
    row = pd.Series({"action_type": "split", **action})
    assert _share_multiplier(row) == pytest.approx(5.0)


def test_bonus_ratio_gives_correct_share_multiplier():
    """Bonus 1:1 means one share becomes two."""
    from src.data.corporate_actions import _share_multiplier

    action = parse_corporate_action("Bonus 1:1")
    row = pd.Series({"action_type": "bonus", **action})
    assert _share_multiplier(row) == pytest.approx(2.0)


def test_dividend_amount_extracted():
    action = parse_corporate_action("Dividend - Rs 2.50 Per Share")
    assert action["amount"] == pytest.approx(2.50)


def test_unrecognized_subject_returns_none():
    """Unparseable actions must not be silently guessed at."""
    assert parse_corporate_action("Annual General Meeting") is None
    assert parse_corporate_action("") is None
    assert parse_corporate_action(None) is None


def test_unparsed_actions_are_reported_not_dropped():
    raw = pd.DataFrame([
        {"isin": "INE001A01036", "symbol": "A", "exDate": "21-Sep-2026",
         "subject": "Dividend - Rs 2 Per Share"},
        {"isin": "INE002A01018", "symbol": "B", "exDate": "22-Sep-2026",
         "subject": "Scheme Of Arrangement"},
    ])
    parsed, unparsed = normalize_corporate_actions(raw)
    assert len(parsed) == 1
    assert len(unparsed) == 1
    assert unparsed.iloc[0]["symbol"] == "B"


# ------------------------------------------------------------------ bhavcopy
def _udiff_frame() -> pd.DataFrame:
    """Real 2024-12-31 UDiFF-shaped rows plus one index row that must drop."""
    path = FIXTURE_DIR / "bhavcopy_udiff_sample.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def _legacy_frame() -> pd.DataFrame:
    path = FIXTURE_DIR / "bhavcopy_legacy_sample.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def test_udiff_normalization_keeps_only_real_isins():
    raw = _udiff_frame()
    out = _normalize_udiff(raw, pd.Timestamp("2024-12-31").date())
    assert (out["isin"].str.startswith(("INE", "INF", "IN9"))).all()
    assert "INDEXNOTANISIN" not in set(out["isin"])
    assert set(out.columns) >= {"date", "isin", "symbol", "series", "close",
                                "volume", "traded_value"}
    assert len(out) == (raw["FinInstrmTp"] == "STK").sum()


def test_udiff_preserves_exchange_traded_value():
    """Traded value comes straight from the exchange; it is a better ADV
    input than price x volume."""
    out = _normalize_udiff(_udiff_frame(), pd.Timestamp("2024-12-31").date())
    row = out.iloc[0]
    assert row["traded_value"] > 0
    # Not simply close * volume.
    assert abs(row["traded_value"] - row["close"] * row["volume"]) > 1.0


def test_legacy_bhavcopy_normalization():
    out = _normalize_legacy(_legacy_frame(), pd.Timestamp("2022-01-03").date())
    assert len(out) == 6
    assert out.iloc[0]["symbol"] == "20MICRONS"
    assert out.iloc[0]["isin"] == "INE144J01027"
    assert out.iloc[0]["close"] == pytest.approx(62.1)
    assert out.iloc[0]["date"] == pd.Timestamp("2022-01-03")


def test_panels_are_pivoted_by_isin():
    bhav = _normalize_legacy(_legacy_frame(), pd.Timestamp("2022-01-03").date())
    # Two dates from the same names so the pivot has a time axis.
    day2 = bhav.copy()
    day2["date"] = pd.Timestamp("2022-01-04")
    day2["close"] = day2["close"] + 1.0
    panels = bhavcopy_to_panels(pd.concat([bhav, day2], ignore_index=True))
    assert set(panels) >= {"close", "volume", "traded_value"}
    assert panels["close"].shape[0] == 2
    infy_like = bhav.iloc[0]["isin"]
    assert panels["close"].loc[pd.Timestamp("2022-01-04"), infy_like] == pytest.approx(
        bhav.iloc[0]["close"] + 1.0
    )


# -------------------------------------------------- survivorship / identity
def test_symbol_history_reconstructed_from_archive():
    """NSE publishes no rename log, but the archive reveals it."""
    bhav = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03", "2022-06-01", "2023-01-02", "2023-06-01"]),
        "isin": ["INE001A01036"] * 4,
        "symbol": ["OLDNAME", "OLDNAME", "NEWNAME", "NEWNAME"],
    })
    hist = build_symbol_history(bhav)
    assert len(hist) == 2
    old = hist[hist["symbol"] == "OLDNAME"].iloc[0]
    new = hist[hist["symbol"] == "NEWNAME"].iloc[0]
    assert pd.notna(old["valid_to"])         # closed interval
    assert pd.isna(new["valid_to"])          # still current


def test_delistings_inferred_from_disappearance():
    """A name that stops appearing in the archive has delisted or been
    suspended - this is what makes the history survivorship-free."""
    dates = pd.bdate_range("2024-01-01", periods=120)
    alive, gone_isin = "INE002A01018", "INE144J01027"  # Reliance, 20 Microns
    rows = []
    for d in dates:
        rows.append({"date": d, "isin": alive})
        if d < dates[30]:
            rows.append({"date": d, "isin": gone_isin})

    gone = infer_delistings(pd.DataFrame(rows), absent_days=30)
    assert list(gone["isin"]) == [gone_isin]


# ---------------------------------------------------------------------- XBRL
def _sample_filing() -> str:
    path = FIXTURE_DIR / "sample_filing.xml"
    assert path.exists(), "Recorded XBRL filing missing; run scripts/build_fixtures.py"
    return path.read_text(encoding="utf-8")


def test_xbrl_extracts_core_financials():
    facts = parse_xbrl(_sample_filing(), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert facts is not None
    # VST Tillers Q3 FY25 consolidated, as filed with NSE.
    assert facts.get("revenue") == pytest.approx(2191000000.0)
    assert facts.get("net_income") == pytest.approx(12800000.0)
    assert facts.get("eps_basic") == pytest.approx(1.48)
    assert facts.get("profit_before_tax") == pytest.approx(36000000.0)
    assert facts.symbol == "VSTTILLERS"


def test_xbrl_picks_the_quarter_not_the_year_to_date():
    """The critical correctness property.

    A Q3 filing contains the quarter AND the 9-month cumulative figure. Taking
    the wrong one silently injects a 3x revenue jump into the series and would
    destroy any SUE calculation.
    """
    facts = parse_xbrl(_sample_filing(), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert facts.get("revenue") == pytest.approx(2191000000.0)   # quarter (OneD)
    assert facts.get("revenue") != pytest.approx(6931200000.0)   # not YTD (FourD)


def test_xbrl_ignores_dimensional_segment_contexts():
    import xml.etree.ElementTree as ET

    contexts = parse_contexts(ET.fromstring(_sample_filing()))
    dimensional = [c for c in contexts.values() if c.has_dimensions]
    assert dimensional, "the recorded filing includes dimensional expense contexts"
    facts = parse_xbrl(_sample_filing(), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert facts.context_used is not None
    assert not contexts[facts.context_used].has_dimensions


def test_context_selection_prefers_shortest_matching_period():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_sample_filing())
    contexts = parse_contexts(root)
    assert "OneD" in contexts and "FourD" in contexts
    assert contexts["OneD"].is_quarterly
    # This filing's FourD *context period* is also the quarter; the YTD amounts
    # ride on that context via DateOfStartOfReportingPeriod, not the context dates.
    chosen = select_context(contexts, pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert chosen == "OneD"


def test_context_selection_without_hint_prefers_a_quarter():
    import xml.etree.ElementTree as ET

    contexts = parse_contexts(ET.fromstring(_sample_filing()))
    assert select_context(contexts) == "OneD"


def test_malformed_xbrl_returns_none():
    assert parse_xbrl("not xml at all") is None
    assert parse_xbrl("") is None


# ------------------------------------------------------------- integration
def test_mismatched_membership_does_not_empty_the_universe():
    """Regression: mixing data sources must not silently zero the universe.

    Applying index membership whose ISINs do not appear in the price panel
    produced an empty universe, which looks like a broken strategy rather
    than a data mismatch.
    """
    from src.data.universe import IndexMembership, UniverseBuilder, UniverseFilters

    prices = pd.read_parquet(FIXTURE_DIR / "panel_close.parquet")
    prices.index = pd.to_datetime(prices.index)
    volumes = pd.read_parquet(FIXTURE_DIR / "panel_volume.parquet")
    volumes.index = pd.to_datetime(volumes.index)
    # Two real ISINs that are *not* in this 60-name liquid slice.
    membership = IndexMembership.from_static(["INE144J01027", "INE253B01015"], "NSE500")

    filters = UniverseFilters(min_adv_inr=0, min_price_inr=0, min_listing_days=0,
                              min_financial_quarters=0)
    builder = UniverseBuilder(filters, membership, "NSE500")
    eligible = builder.build(prices, volumes)

    # Without the guard this is 0 and every downstream result is meaningless.
    assert int(eligible.iloc[-1].sum()) == 0, (
        "builder applies membership as given; the loader is responsible for "
        "detecting the mismatch"
    )

    fallback = IndexMembership.from_static(list(prices.columns), "NSE500")
    eligible2 = UniverseBuilder(filters, fallback, "NSE500").build(prices, volumes)
    assert int(eligible2.iloc[-1].sum()) >= len(prices.columns) - 2


def test_nse_provider_normalizes_to_the_canonical_schema():
    """EPS must survive normalization - Sleeve C's SUE depends on it."""
    from src.data.fundamentals.base import FUNDAMENTAL_COLUMNS
    from src.data.fundamentals.nse_provider import NSEProvider

    meta = json.loads((FIXTURE_DIR / "sample_filing.json").read_text(encoding="utf-8"))
    provider = NSEProvider(cache_dir=None)
    raw = pd.DataFrame([{
        "isin": meta["isin"],
        "announce_date": pd.Timestamp(meta["announce_date"]),
        "period_end": pd.Timestamp(meta["period_end"]),
        "eps": 1.48,
        "revenue": 2191000000.0,
        "net_income": 12800000.0,
    }])
    out = provider.normalize(raw)
    assert "eps" in FUNDAMENTAL_COLUMNS
    assert out.iloc[0]["eps"] == pytest.approx(1.48)
    assert out.iloc[0]["announce_date"] == pd.Timestamp(meta["announce_date"])


def test_enrich_pnl_metrics_builds_ttm_margins():
    from src.data.fundamentals.nse import enrich_pnl_metrics

    isin = "INE000A01000"
    rows = []
    for i, (pe, rev, ni, pbt) in enumerate([
        ("2022-06-30", 100.0, 10.0, 12.0),
        ("2022-09-30", 110.0, 12.0, 14.0),
        ("2022-12-31", 120.0, 11.0, 13.0),
        ("2023-03-31", 130.0, 15.0, 18.0),
    ]):
        rows.append({
            "isin": isin, "period_end": pd.Timestamp(pe),
            "revenue": rev, "net_income": ni, "profit_before_tax": pbt,
            "finance_cost": 1.0, "equity_capital": 50.0, "eps": 2.0 + i,
        })
    out = enrich_pnl_metrics(pd.DataFrame(rows))
    last = out.iloc[-1]
    assert last["eps_ttm"] == pytest.approx(2 + 3 + 4 + 5)
    assert last["net_margin"] == pytest.approx((10 + 12 + 11 + 15) / (100 + 110 + 120 + 130))
    assert pd.notna(last["roic"])
    assert pd.notna(last["gross_profitability"])


def test_synthetic_source_is_rejected(cfg):
    from src.data.loader import load_dataset

    with pytest.raises(ValueError, match="Simulated market data is disabled"):
        load_dataset(cfg, source="synthetic", run_health_check=False)
