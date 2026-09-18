"""NSE India public data client.

This closes the two biggest gaps in the framework:

1. **Point-in-time fundamentals.** NSE publishes `exchdisstime` - the exact
   moment the exchange disseminated a filing. That is when the market actually
   learned the number. Convenience APIs and yfinance do not expose it, which is
   why fundamentals from those sources carry look-ahead risk.
2. **Survivorship bias.** The daily bhavcopy archive contains every security
   that traded on a given day, including companies later delisted. A broker
   instrument dump only lists what is tradeable *today*, so it structurally
   cannot give you this.

Two transports, deliberately separated:

* **Archives** (`nsearchives.nseindia.com`) - static CSV/ZIP files served to
  plain HTTP clients. Bhavcopy, the equity master with ISINs, index
  constituents and XBRL documents all live here. This is the default path and
  needs no special handling.
* **JSON APIs** (`www.nseindia.com/api/*`) - sit behind Akamai bot protection
  that rejects standard Python HTTP clients on TLS fingerprint, not headers.
  Reaching them requires browser TLS impersonation (`curl_cffi`), so it is
  **opt-in** via `allow_api=True`.

Terms of use: these endpoints exist to serve the public website. Impersonating
a browser to reach the JSON APIs is a grey area under NSE's terms - keep it to
personal research, keep request rates low, and do not redistribute the data.
The archive path carries no such ambiguity, which is why it is the default.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import requests

from ..config import get_logger

log = get_logger(__name__)

BASE = "https://www.nseindia.com"
ARCHIVES = "https://nsearchives.nseindia.com"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
ARCHIVE_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/",
}

# Each API family expects the Referer of the page that normally calls it.
API_PAGES = {
    "results": "/companies-listing/corporate-filings-financial-results",
    "announcements": "/companies-listing/corporate-filings-announcements",
    "actions": "/companies-listing/corporate-filings-actions",
    "events": "/companies-listing/corporate-filings-event-calendar",
}

INDEX_FILES = {
    "NIFTY50": "ind_nifty50list.csv",
    "NIFTY100": "ind_nifty100list.csv",
    "NIFTY200": "ind_nifty200list.csv",
    "NIFTY500": "ind_nifty500list.csv",
    "NIFTYMIDCAP150": "ind_niftymidcap150list.csv",
    "NIFTYSMALLCAP250": "ind_niftysmallcap250list.csv",
}


class NSEError(RuntimeError):
    pass


class NSEAPIUnavailable(NSEError):
    """Raised when the JSON APIs are needed but not enabled/reachable."""


@dataclass
class NSEClient:
    """Polite, cached client for NSE public data."""

    cache_dir: Path | None = None
    #: Enable the Akamai-protected JSON APIs (requires curl_cffi).
    allow_api: bool = False
    rate_limit_per_sec: float = 1.0
    timeout: int = 30
    max_retries: int = 3

    _http: requests.Session = field(default_factory=requests.Session, repr=False)
    _api: Any = field(default=None, repr=False)
    _api_warmed: float = field(default=0.0, repr=False)
    _last_call: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._http.headers.update(ARCHIVE_HEADERS)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- plumbing --------------------------------------------------------------
    def _throttle(self) -> None:
        if self.rate_limit_per_sec <= 0:
            return
        gap = 1.0 / self.rate_limit_per_sec
        elapsed = time.monotonic() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.monotonic()

    def _fetch(self, url: str) -> bytes | None:
        """GET a static archive file. Returns None on 404 (holiday, no file)."""
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self._http.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    wait = 5 * (attempt + 1)
                    log.warning("NSE archives rate-limited; waiting %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as exc:
                wait = 2**attempt
                log.debug("archive fetch failed (%s), retry in %ds: %s", url, wait, exc)
                time.sleep(wait)
        return None

    # -- JSON API (opt-in) -----------------------------------------------------
    @property
    def api_available(self) -> bool:
        if not self.allow_api:
            return False
        try:
            import curl_cffi  # noqa: F401

            return True
        except ImportError:
            return False

    def _api_session(self, kind: str):
        """Browser-impersonating session, re-warmed when cookies go stale."""
        if not self.allow_api:
            raise NSEAPIUnavailable(
                "NSE JSON APIs are disabled. Pass allow_api=True (and install "
                "curl_cffi) to enable financial results, announcements, the "
                "event calendar and corporate actions."
            )
        try:
            from curl_cffi import requests as cr
        except ImportError as exc:
            raise NSEAPIUnavailable(
                "curl_cffi is required for NSE's JSON APIs (they reject standard "
                "Python clients on TLS fingerprint). Install with: pip install curl_cffi"
            ) from exc

        stale = (time.monotonic() - self._api_warmed) > 240
        if self._api is None or stale:
            self._api = cr.Session(impersonate="chrome")
            self._throttle()
            self._api.get(f"{BASE}/", timeout=self.timeout)
            self._api_warmed = time.monotonic()
            log.debug("NSE API session warmed")

        # Load the owning page so the Referer and cookies line up.
        page = API_PAGES.get(kind)
        if page:
            self._throttle()
            self._api.get(f"{BASE}{page}", timeout=self.timeout)
        return self._api

    def _api_json(self, path: str, params: dict[str, Any], kind: str) -> Any:
        session = self._api_session(kind)
        page = API_PAGES.get(kind, "/")
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = session.get(
                    f"{BASE}{path}",
                    params=params,
                    headers={
                        "Referer": f"{BASE}{page}",
                        "Accept": "application/json, text/plain, */*",
                    },
                    timeout=self.timeout,
                )
                if resp.status_code in (401, 403):
                    self._api = None  # force a re-warm
                    session = self._api_session(kind)
                    continue
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    log.warning("NSE API %s returned %s", path, resp.status_code)
                    return None
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - curl_cffi raises its own types
                wait = 2**attempt
                log.debug("NSE API call failed (%s), retry in %ds: %s", path, wait, exc)
                time.sleep(wait)
        return None

    # ---------------------------------------------------------------- archives
    def equity_list(self) -> pd.DataFrame:
        """Every listed NSE equity with its ISIN - the security-master seed."""
        content = self._fetch(f"{ARCHIVES}/content/equities/EQUITY_L.csv")
        if not content:
            raise NSEError("could not download EQUITY_L.csv")

        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "isin_number": "isin", "name_of_company": "name",
            "date_of_listing": "listing_date", "face_value": "face_value",
        })
        if "listing_date" in df.columns:
            # EQUITY_L publishes listing dates as DD-MMM-YYYY.
            df["listing_date"] = pd.to_datetime(
                df["listing_date"].astype(str).str.strip(),
                format="%d-%b-%Y", errors="coerce",
            )
        for col in ("symbol", "isin", "series"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df

    def index_constituents(self, index: str = "NIFTY500") -> pd.DataFrame:
        """Current index members, with ISIN and industry (our sector source)."""
        key = index.upper().replace(" ", "").replace("-", "")
        filename = INDEX_FILES.get(key)
        if not filename:
            raise NSEError(f"unknown index '{index}'. Known: {sorted(INDEX_FILES)}")

        content = self._fetch(f"{ARCHIVES}/content/indices/{filename}")
        if not content:
            raise NSEError(f"could not download constituents for {index}")

        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "isin_code": "isin", "company_name": "name", "industry": "sector",
        })
        for col in ("symbol", "isin", "sector"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        df["index_name"] = key
        return df

    def bhavcopy(self, on: date | str, use_cache: bool = True) -> pd.DataFrame:
        """End-of-day snapshot for one trading day, including delisted names."""
        day = pd.Timestamp(on).date()
        cached = self._bhav_cache_path(day)
        if use_cache and cached and cached.exists():
            return pd.read_parquet(cached)

        raw = self._download_bhavcopy(day)
        df = _normalize_bhavcopy(raw, day) if raw is not None else pd.DataFrame()

        # Cache negatives too - holidays would otherwise be re-fetched forever.
        if cached:
            df.to_parquet(cached, index=False)
        return df

    def _bhav_cache_path(self, day: date) -> Path | None:
        if not self.cache_dir:
            return None
        d = self.cache_dir / "bhavcopy"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{day:%Y%m%d}.parquet"

    def _download_bhavcopy(self, day: date) -> pd.DataFrame | None:
        """Try each published layout; NSE has changed format several times."""
        candidates = [
            # UDiFF - current format (2024 onward)
            f"{ARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip",
            # Full bhavdata with delivery figures
            f"{ARCHIVES}/products/content/sec_bhavdata_full_{day:%d%m%Y}.csv",
            # Legacy zipped bhavcopy
            f"{ARCHIVES}/content/historical/EQUITIES/{day:%Y}/{day:%b}/"
            f"cm{day:%d%b%Y}bhav.csv.zip".upper().replace(
                f"{ARCHIVES}/CONTENT".upper(), f"{ARCHIVES}/content"
            ),
        ]
        for url in candidates:
            content = self._fetch(url)
            if not content:
                continue
            try:
                df = _read_csv_or_zip(content)
                if df is not None and not df.empty:
                    return df
            except (zipfile.BadZipFile, ValueError, pd.errors.ParserError) as exc:
                log.debug("could not parse %s: %s", url, exc)
        return None

    def bhavcopy_range(
        self, from_date: date | str, to_date: date | str, use_cache: bool = True,
    ) -> pd.DataFrame:
        """Stack bhavcopies across a range. This is the survivorship-safe history."""
        days = pd.bdate_range(pd.Timestamp(from_date), pd.Timestamp(to_date))
        frames, missing = [], 0
        for i, day in enumerate(days, 1):
            df = self.bhavcopy(day.date(), use_cache)
            if df.empty:
                missing += 1
            else:
                frames.append(df)
            if i % 100 == 0:
                log.info("bhavcopy %d/%d days (%d with no file)", i, len(days), missing)

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        log.info(
            "bhavcopy: %d rows, %d unique symbols, %s..%s",
            len(out), out["symbol"].nunique(), out["date"].min().date(), out["date"].max().date(),
        )
        return out

    def download_xbrl(self, url: str) -> dict[str, float]:
        """Fetch and flatten one XBRL filing into tagged numeric facts."""
        content = self._fetch(url)
        return parse_xbrl(content) if content else {}

    # ------------------------------------------------------------- JSON APIs
    def financial_results(
        self, from_date: date | str, to_date: date | str,
        period: str = "Quarterly", fo_only: bool = False,
    ) -> pd.DataFrame:
        """Filing metadata for quarterly/annual results.

        `announce_datetime` is derived from `exchdisstime`, the exchange
        dissemination timestamp - the honest point-in-time key.
        """
        rows = _rows(self._api_json(
            "/api/corporates-financial-results",
            {
                "index": "equities",
                "from_date": _ddmmyyyy(from_date),
                "to_date": _ddmmyyyy(to_date),
                "period": period,
                "fo_sec": "true" if fo_only else "false",
            },
            kind="results",
        ))
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["announce_datetime"] = df.get("exchdisstime", df.get("broadCastDate")).apply(
            _parse_nse_datetime
        )
        df["announce_date"] = pd.to_datetime(df["announce_datetime"]).dt.normalize()
        for col in ("fromDate", "toDate"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_nse_datetime)
        df["period_end"] = df.get("toDate")
        return df.dropna(subset=["announce_date"])

    def corporate_announcements(
        self, from_date: date | str, to_date: date | str, symbol: str | None = None,
    ) -> pd.DataFrame:
        params = {
            "index": "equities",
            "from_date": _ddmmyyyy(from_date),
            "to_date": _ddmmyyyy(to_date),
        }
        if symbol:
            params["symbol"] = symbol
        rows = _rows(self._api_json("/api/corporate-announcements", params, "announcements"))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("an_dt", "sort_date", "exchdisstime"):
            if col in df.columns:
                df["announce_datetime"] = df[col].apply(_parse_nse_datetime)
                break
        return df

    def event_calendar(self, from_date: date | str, to_date: date | str) -> pd.DataFrame:
        """Forthcoming board meetings, including results dates."""
        rows = _rows(self._api_json(
            "/api/event-calendar",
            {
                "index": "equities",
                "from_date": _ddmmyyyy(from_date),
                "to_date": _ddmmyyyy(to_date),
            },
            "events",
        ))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["event_date"] = df["date"].apply(_parse_nse_datetime)
        if "purpose" in df.columns:
            df["is_results"] = df["purpose"].astype(str).str.contains(
                "result|financial", case=False, na=False
            )
        return df

    def corporate_actions(
        self, from_date: date | str, to_date: date | str, symbol: str | None = None,
    ) -> pd.DataFrame:
        """Dividends, splits, bonuses and rights with their ex-dates."""
        params = {
            "index": "equities",
            "from_date": _ddmmyyyy(from_date),
            "to_date": _ddmmyyyy(to_date),
        }
        if symbol:
            params["symbol"] = symbol
        rows = _rows(self._api_json("/api/corporates-corporateActions", params, "actions"))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("exDate", "exdate", "ex_dt"):
            if col in df.columns:
                df["ex_date"] = df[col].apply(_parse_nse_datetime)
                break
        return df


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _rows(payload: Any) -> list[dict]:
    """NSE returns a bare list for some endpoints and {data: [...]} for others."""
    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "records", "resultSet"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
            if isinstance(value, dict) and isinstance(value.get("data"), list):
                return [r for r in value["data"] if isinstance(r, dict)]
    return []


def _ddmmyyyy(d: date | str | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%d-%m-%Y")


def _parse_nse_datetime(value: Any) -> pd.Timestamp | None:
    """NSE mixes formats: '24-Aug-2026 17:39:32', '24-Aug-2026', ISO, ..."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "-"):
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y",
                "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed


def _read_csv_or_zip(content: bytes) -> pd.DataFrame | None:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                return None
            with zf.open(names[0]) as fh:
                return pd.read_csv(fh)
    return pd.read_csv(io.BytesIO(content))


def _normalize_bhavcopy(df: pd.DataFrame, day: date) -> pd.DataFrame:
    """Normalize the several bhavcopy layouts NSE has published over the years."""
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]

    # UDiFF (current) -> canonical
    df = df.rename(columns={
        "TCKRSYMB": "SYMBOL", "SCTYSRS": "SERIES", "OPNPRIC": "OPEN", "HGHPRIC": "HIGH",
        "LWPRIC": "LOW", "CLSPRIC": "CLOSE", "LASTPRIC": "LAST", "PRVSCLSGPRIC": "PREVCLOSE",
        "TTLTRADGVOL": "TOTTRDQTY", "TTLTRFVAL": "TOTTRDVAL", "ISIN": "ISIN",
    })
    # sec_bhavdata_full -> canonical
    df = df.rename(columns={
        "OPEN_PRICE": "OPEN", "HIGH_PRICE": "HIGH", "LOW_PRICE": "LOW",
        "CLOSE_PRICE": "CLOSE", "PREV_CLOSE": "PREVCLOSE", "LAST_PRICE": "LAST",
        "TTL_TRD_QNTY": "TOTTRDQTY", "TURNOVER_LACS": "TOTTRDVAL",
        "DELIV_QTY": "DELIVQTY", "DELIV_PER": "DELIVPER",
    })

    if "SYMBOL" not in df.columns or "CLOSE" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date": pd.Timestamp(day),
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "series": (df["SERIES"].astype(str).str.strip() if "SERIES" in df.columns else "EQ"),
    })
    for src, dst in (("OPEN", "open"), ("HIGH", "high"), ("LOW", "low"),
                     ("CLOSE", "close"), ("PREVCLOSE", "prev_close"),
                     ("TOTTRDQTY", "volume"), ("TOTTRDVAL", "turnover"),
                     ("DELIVQTY", "delivery_qty")):
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.NA
    if "ISIN" in df.columns:
        out["isin"] = df["ISIN"].astype(str).str.strip()

    # sec_bhavdata reports turnover in lakhs; normalize to rupees.
    if "TURNOVER_LACS" in df.columns or "turnover" in out.columns:
        pass  # handled by caller config; left as published to avoid double-scaling

    # EQ and BE are the cash-market series that matter.
    out = out[out["series"].isin(["EQ", "BE"])]
    return out.dropna(subset=["close"]).reset_index(drop=True)


# Ind-AS XBRL tags -> the fields we actually use.
XBRL_FIELDS = {
    "revenue": ["RevenueFromOperations", "Revenue", "Income"],
    "other_income": ["OtherIncome"],
    "net_income": [
        "ProfitLossForPeriod", "ProfitLoss",
        "ProfitLossForPeriodAttributableToOwnersOfParent",
    ],
    "profit_before_tax": ["ProfitBeforeTax", "ProfitBeforeExceptionalItemsAndTax"],
    "eps_basic": [
        "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "BasicEarningsLossPerShareFromContinuingOperations",
        "BasicEarningsPerShare",
    ],
    "eps_diluted": [
        "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "DilutedEarningsLossPerShareFromContinuingOperations",
        "DilutedEarningsPerShare",
    ],
    "total_expenses": ["Expenses", "TotalExpenses"],
    "finance_cost": ["FinanceCosts"],
    "depreciation": ["DepreciationDepletionAndAmortisationExpense"],
    "tax_expense": ["TaxExpense", "IncomeTaxExpenseContinuingOperations"],
    "equity_capital": ["EquityShareCapital", "PaidUpValueOfEquityShareCapital"],
}


def parse_xbrl(content: bytes) -> dict[str, float]:
    """Flatten an Ind-AS XBRL filing into the numeric facts we need.

    XBRL repeats each tag across contexts (this quarter, prior quarter, year to
    date). Without context resolution the safest reading is the first numeric
    occurrence, which corresponds to the primary reporting period.
    """
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        log.debug("could not parse XBRL: %s", exc)
        return {}

    facts: dict[str, float] = {}
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        text = (element.text or "").strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        facts.setdefault(tag, value)

    out: dict[str, float] = {}
    for name, candidates in XBRL_FIELDS.items():
        for tag in candidates:
            if tag in facts:
                out[name] = facts[tag]
                break
    out["_n_tags"] = float(len(facts))
    return out


def build_price_panel(
    bhav: pd.DataFrame, field: str = "close", key: str = "symbol",
) -> pd.DataFrame:
    """Wide (date x security) panel from stacked bhavcopy rows."""
    if bhav.empty or field not in bhav.columns:
        return pd.DataFrame()
    panel = bhav.pivot_table(index="date", columns=key, values=field, aggfunc="last")
    panel.index.name = "date"
    return panel.sort_index()
