"""NSE public data client.

NSE publishes, for free, most of what this framework previously needed a paid
vendor for:

  * ``EQUITY_L.csv``        - real ISINs, listing dates, face value
  * **bhavcopy archives**   - daily OHLCV for the *entire* market, ISIN-keyed.
    This is the important one: because every historical file lists whatever
    was trading that day, replaying the archive reconstructs a
    survivorship-bias-free database including companies that later delisted.
  * corporate actions       - splits, bonuses, dividends (with ex-dates)
  * board meetings          - forthcoming results dates
  * financial results       - with ``broadCastDate``, the exact announcement
    timestamp that point-in-time discipline depends on, plus XBRL links
    carrying revenue, PAT and EPS
  * index constituent files - NSE 500 membership and industry classification

Operational notes learned the hard way:
  * ``www.nseindia.com`` rejects unprimed requests (403). The ``nsearchives``
    host does not, so archive downloads work without a session.
  * The JSON APIs need browser-like headers and, sometimes, cookies from the
    homepage. We prime once and degrade gracefully.
  * Be polite: this is a public service, not a paid API. Requests are
    rate-limited and cached on disk.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from ..config import get_logger

log = get_logger(__name__)

ARCHIVES = "https://nsearchives.nseindia.com"
WWW = "https://www.nseindia.com"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Series we treat as ordinary equity. BE/BZ are surveillance/trade-to-trade
# segments; SM/ST are SME boards - all excluded from the investable universe.
EQUITY_SERIES = {"EQ"}

INDEX_FILES = {
    "NSE500": "ind_nifty500list.csv",
    "NIFTY500": "ind_nifty500list.csv",
    "NIFTY200": "ind_nifty200list.csv",
    "NIFTY100": "ind_nifty100list.csv",
    "NIFTY50": "ind_nifty50list.csv",
    "NIFTYMIDCAP150": "ind_niftymidcap150list.csv",
    "NIFTYSMALLCAP250": "ind_niftysmallcap250list.csv",
}


class NSEClient:
    """Polite, cached HTTP client for NSE public data."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        rate_limit_per_sec: float = 2.0,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.cache_dir = Path(cache_dir) / "nse" if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = timeout
        self.max_retries = max_retries
        self._min_gap = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.0
        self._last_call = 0.0
        self._primed = False

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{WWW}/",
        })

    # -- plumbing --------------------------------------------------------------
    def _throttle(self) -> None:
        if self._min_gap <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)
        self._last_call = time.monotonic()

    def _prime(self) -> None:
        """Fetch cookies from the homepage; the JSON APIs often require them."""
        if self._primed:
            return
        try:
            self._throttle()
            self.session.get(WWW, timeout=self.timeout)
        except requests.RequestException as exc:
            log.debug("NSE cookie priming failed (archives still work): %s", exc)
        self._primed = True

    def _get(self, url: str, prime: bool = False) -> requests.Response | None:
        if prime:
            self._prime()
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 404:
                    return None          # genuinely absent (holiday, bad date)
                if resp.status_code in (401, 403) and not prime:
                    self._primed = False
                    self._prime()
                log.debug("NSE %s -> %s (attempt %d)", url, resp.status_code, attempt + 1)
            except requests.RequestException as exc:
                log.debug("NSE request failed (%s): %s", url, exc)
            time.sleep(2 ** attempt)
        log.warning("Giving up on %s", url)
        return None

    def _cache_path(self, name: str) -> Path | None:
        return self.cache_dir / name if self.cache_dir else None

    # -- security master -------------------------------------------------------
    def fetch_equity_list(self, refresh: bool = False) -> pd.DataFrame:
        """All listed NSE equities with real ISINs and listing dates."""
        cache = self._cache_path("equity_list.parquet")
        if cache and cache.exists() and not refresh:
            return pd.read_parquet(cache)

        resp = self._get(f"{ARCHIVES}/content/equities/EQUITY_L.csv")
        if resp is None:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "name": df["NAME_OF_COMPANY"].astype(str).str.strip(),
            "series": df["SERIES"].astype(str).str.strip(),
            "listing_date": pd.to_datetime(df["DATE_OF_LISTING"], format="%d-%b-%Y", errors="coerce"),
            "isin": df["ISIN_NUMBER"].astype(str).str.strip(),
            "face_value": pd.to_numeric(df.get("FACE_VALUE"), errors="coerce"),
        })
        out = out[out["isin"].str.match(r"^INE|^INF|^IN9", na=False)]

        if cache:
            out.to_parquet(cache, index=False)
        log.info("NSE equity list: %d instruments with ISINs", len(out))
        return out

    def fetch_index_constituents(self, index: str = "NSE500", refresh: bool = False) -> pd.DataFrame:
        """Current index members, including NSE's industry classification."""
        filename = INDEX_FILES.get(index.upper().replace(" ", ""))
        if not filename:
            log.warning("Unknown index '%s'; known: %s", index, sorted(INDEX_FILES))
            return pd.DataFrame()

        cache = self._cache_path(f"constituents_{index}.parquet")
        if cache and cache.exists() and not refresh:
            return pd.read_parquet(cache)

        resp = self._get(f"{ARCHIVES}/content/indices/{filename}")
        if resp is None:
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        out = pd.DataFrame({
            "symbol": df.get("symbol", pd.Series(dtype=str)).astype(str).str.strip(),
            "name": df.get("company_name", pd.Series(dtype=str)).astype(str).str.strip(),
            "sector": df.get("industry", pd.Series(dtype=str)).astype(str).str.strip(),
            "isin": df.get("isin_code", pd.Series(dtype=str)).astype(str).str.strip(),
            "index_name": index.upper(),
        })
        if cache:
            out.to_parquet(cache, index=False)
        log.info("%s constituents: %d names", index, len(out))
        return out

    # -- prices ----------------------------------------------------------------
    def fetch_bhavcopy(self, on: date, refresh: bool = False) -> pd.DataFrame:
        """One day's full-market OHLCV, ISIN-keyed.

        Tries the modern UDiFF layout first and falls back to the legacy
        format for older dates. Returns an empty frame on holidays.
        """
        on = pd.Timestamp(on).date()
        cache = self._cache_path(f"bhav_{on:%Y%m%d}.parquet")
        if cache and cache.exists() and not refresh:
            return pd.read_parquet(cache)

        df = self._bhavcopy_udiff(on)
        if df.empty:
            df = self._bhavcopy_legacy(on)

        if not df.empty and cache:
            df.to_parquet(cache, index=False)
        return df

    def _bhavcopy_udiff(self, on: date) -> pd.DataFrame:
        url = f"{ARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{on:%Y%m%d}_F_0000.csv.zip"
        resp = self._get(url)
        if resp is None:
            return pd.DataFrame()
        try:
            raw = _read_zipped_csv(resp.content)
        except (zipfile.BadZipFile, ValueError) as exc:
            log.debug("UDiFF bhavcopy unreadable for %s: %s", on, exc)
            return pd.DataFrame()
        return _normalize_udiff(raw, on)

    def _bhavcopy_legacy(self, on: date) -> pd.DataFrame:
        month = f"{on:%b}".upper()
        url = (f"{ARCHIVES}/content/historical/EQUITIES/{on:%Y}/{month}/"
               f"cm{on:%d}{month}{on:%Y}bhav.csv.zip")
        resp = self._get(url)
        if resp is None:
            return pd.DataFrame()
        try:
            raw = _read_zipped_csv(resp.content)
        except (zipfile.BadZipFile, ValueError):
            return pd.DataFrame()
        return _normalize_legacy(raw, on)

    def fetch_bhavcopy_range(
        self,
        start: date | str,
        end: date | str,
        series: Iterable[str] | None = None,
        progress_every: int = 25,
    ) -> pd.DataFrame:
        """Replay the archive over a date range.

        This is how the survivorship-bias-free price history is built: each
        file contains exactly the securities trading that day, so names that
        later delisted remain in the record.
        """
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        keep = set(series) if series else EQUITY_SERIES

        frames, fetched, skipped = [], 0, 0
        current = start
        while current <= end:
            if current.weekday() < 5:          # skip weekends outright
                df = self.fetch_bhavcopy(current)
                if df.empty:
                    skipped += 1               # holiday or missing file
                else:
                    if keep and "series" in df.columns:
                        df = df[df["series"].isin(keep)]
                    frames.append(df)
                    fetched += 1
                    if progress_every and fetched % progress_every == 0:
                        log.info("bhavcopy: %d days fetched (at %s)", fetched, current)
            current += timedelta(days=1)

        if not frames:
            log.warning("No bhavcopy data between %s and %s", start, end)
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        log.info("bhavcopy range: %d days, %d rows, %d unique ISINs (%d non-trading days)",
                 fetched, len(out), out["isin"].nunique(), skipped)
        return out

    # -- corporate actions -----------------------------------------------------
    def fetch_corporate_actions(self, refresh: bool = False) -> pd.DataFrame:
        """Recent/forthcoming splits, bonuses and dividends."""
        cache = self._cache_path("corporate_actions_raw.parquet")
        resp = self._get(f"{WWW}/api/corporates-corporateActions?index=equities", prime=True)
        if resp is None:
            if cache and cache.exists():
                log.info("Using cached corporate actions")
                return pd.read_parquet(cache)
            return pd.DataFrame()

        try:
            rows = resp.json()
        except ValueError:
            return pd.DataFrame()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if cache:
            df.to_parquet(cache, index=False)
        return df

    # -- earnings --------------------------------------------------------------
    def fetch_board_meetings(self) -> pd.DataFrame:
        """Forthcoming board meetings - the forward-looking results calendar."""
        resp = self._get(f"{WWW}/api/corporate-board-meetings?index=equities", prime=True)
        if resp is None:
            return pd.DataFrame()
        try:
            rows = resp.json()
        except ValueError:
            return pd.DataFrame()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "symbol": df.get("bm_symbol"),
            "isin": df.get("sm_isin"),
            "name": df.get("sm_name"),
            "meeting_date": pd.to_datetime(df.get("bm_date"), format="%d-%b-%Y", errors="coerce"),
            "purpose": df.get("bm_purpose"),
            "description": df.get("bm_desc"),
            "intimated_at": pd.to_datetime(df.get("bm_timestamp"), format="%d-%b-%Y %H:%M:%S",
                                           errors="coerce"),
        })
        # Only meetings that actually concern financial results.
        text = (out["purpose"].fillna("") + " " + out["description"].fillna("")).str.lower()
        out["is_results"] = text.str.contains("financial result|quarterly result|audited result")
        return out.dropna(subset=["meeting_date"])

    def fetch_financial_results(
        self, period: str = "Quarterly", refresh: bool = False
    ) -> pd.DataFrame:
        """Results filings with announcement timestamps and XBRL links.

        ``broadCastDate`` is the field that makes point-in-time analysis
        possible: it is when the market actually learned the numbers.
        """
        cache = self._cache_path(f"financial_results_{period}.parquet")
        resp = self._get(
            f"{WWW}/api/corporates-financial-results?index=equities&period={period}", prime=True
        )
        if resp is None:
            if cache and cache.exists():
                log.info("Using cached financial results")
                return pd.read_parquet(cache)
            return pd.DataFrame()

        try:
            rows = resp.json()
        except ValueError:
            return pd.DataFrame()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "symbol": df.get("symbol"),
            "isin": df.get("isin"),
            "company": df.get("companyName"),
            "period_start": _parse_date(df.get("fromDate")),
            "period_end": _parse_date(df.get("toDate")),
            "announce_date": _parse_datetime(df.get("broadCastDate")),
            "filing_date": _parse_datetime(df.get("filingDate"), "%d-%b-%Y %H:%M"),
            "relating_to": df.get("relatingTo"),
            "consolidated": df.get("consolidated"),
            "audited": df.get("audited"),
            "period": df.get("period"),
            "xbrl_url": df.get("xbrl"),
        })
        out = out.dropna(subset=["isin", "period_end"])
        # Fall back to the filing timestamp when broadcast is missing.
        out["announce_date"] = out["announce_date"].fillna(out["filing_date"])

        if cache:
            out.to_parquet(cache, index=False)
        log.info("NSE financial results: %d filings", len(out))
        return out

    def fetch_xbrl(self, url: str) -> str | None:
        """Download one XBRL results document."""
        if not isinstance(url, str) or not url.startswith("http"):
            return None
        resp = self._get(url)
        return resp.text if resp is not None else None


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _read_zipped_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("no CSV inside the archive")
        with z.open(names[0]) as fh:
            return pd.read_csv(fh)


def _normalize_udiff(raw: pd.DataFrame, on: date) -> pd.DataFrame:
    """Modern UDiFF bhavcopy -> canonical columns."""
    if raw.empty or "TckrSymb" not in raw.columns:
        return pd.DataFrame()

    df = raw.copy()
    if "FinInstrmTp" in df.columns:
        df = df[df["FinInstrmTp"].astype(str).str.upper() == "STK"]

    out = pd.DataFrame({
        "date": pd.to_datetime(df.get("TradDt"), errors="coerce"),
        "isin": df.get("ISIN").astype(str).str.strip(),
        "symbol": df.get("TckrSymb").astype(str).str.strip(),
        "series": df.get("SctySrs").astype(str).str.strip(),
        "open": pd.to_numeric(df.get("OpnPric"), errors="coerce"),
        "high": pd.to_numeric(df.get("HghPric"), errors="coerce"),
        "low": pd.to_numeric(df.get("LwPric"), errors="coerce"),
        "close": pd.to_numeric(df.get("ClsPric"), errors="coerce"),
        "prev_close": pd.to_numeric(df.get("PrvsClsgPric"), errors="coerce"),
        "volume": pd.to_numeric(df.get("TtlTradgVol"), errors="coerce"),
        # Traded value straight from the exchange beats price x volume.
        "traded_value": pd.to_numeric(df.get("TtlTrfVal"), errors="coerce"),
        "trades": pd.to_numeric(df.get("TtlNbOfTxsExctd"), errors="coerce"),
    })
    out["date"] = out["date"].fillna(pd.Timestamp(on))
    return out[out["isin"].str.startswith(("INE", "INF", "IN9"), na=False)].reset_index(drop=True)


def _normalize_legacy(raw: pd.DataFrame, on: date) -> pd.DataFrame:
    """Pre-2024 bhavcopy layout -> canonical columns."""
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date": pd.to_datetime(df.get("TIMESTAMP"), format="%d-%b-%Y", errors="coerce"),
        "isin": df.get("ISIN", pd.Series(dtype=str)).astype(str).str.strip(),
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "series": df.get("SERIES", pd.Series(dtype=str)).astype(str).str.strip(),
        "open": pd.to_numeric(df.get("OPEN"), errors="coerce"),
        "high": pd.to_numeric(df.get("HIGH"), errors="coerce"),
        "low": pd.to_numeric(df.get("LOW"), errors="coerce"),
        "close": pd.to_numeric(df.get("CLOSE"), errors="coerce"),
        "prev_close": pd.to_numeric(df.get("PREVCLOSE"), errors="coerce"),
        "volume": pd.to_numeric(df.get("TOTTRDQTY"), errors="coerce"),
        "traded_value": pd.to_numeric(df.get("TOTTRDVAL"), errors="coerce"),
        "trades": pd.to_numeric(df.get("TOTALTRADES"), errors="coerce"),
    })
    out["date"] = out["date"].fillna(pd.Timestamp(on))
    return out[out["isin"].str.startswith(("INE", "INF", "IN9"), na=False)].reset_index(drop=True)


def _parse_date(s: Any) -> pd.Series:
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")


def _parse_datetime(s: Any, fmt: str = "%d-%b-%Y %H:%M:%S") -> pd.Series:
    if s is None:
        return pd.Series(dtype="datetime64[ns]")
    out = pd.to_datetime(s, format=fmt, errors="coerce")
    missing = out.isna()
    if missing.any():
        out[missing] = pd.to_datetime(pd.Series(s)[missing], errors="coerce", dayfirst=True)
    return out


# --------------------------------------------------------------------------- #
# Corporate action subject parsing
# --------------------------------------------------------------------------- #
_DIVIDEND = re.compile(r"divid", re.I)
# NSE writes currency as Rs / Re / Rupee(s) / INR, often with a trailing "/-".
# "Re 1" (singular) is common for a one-rupee face value, so the pattern has
# to accept it - missing it would mean missing a split entirely.
_CCY = r"(?:rs|re|rupees?|inr)\.?\s*"
_SPLIT = re.compile(
    rf"split.*?(?:from\s*)?{_CCY}?(\d+(?:\.\d+)?)\s*(?:/-)?\s*to\s*{_CCY}?(\d+(?:\.\d+)?)",
    re.I | re.S,
)
_BONUS = re.compile(r"bonus.*?(\d+)\s*[:/]\s*(\d+)", re.I | re.S)
_AMOUNT = re.compile(rf"{_CCY}(\d+(?:\.\d+)?)", re.I)


def parse_corporate_action(subject: str) -> dict[str, Any] | None:
    """Turn NSE's free-text ``subject`` into a structured action.

    Examples seen in the wild:
        "Dividend - Rs 2 Per Share"
        "Face Value Split From Rs 10 To Rs 2"
        "Bonus 1:1"

    A missed split silently corrupts every return calculation for that name,
    so anything unrecognised is returned as None and reported rather than
    guessed at.
    """
    if not isinstance(subject, str) or not subject.strip():
        return None
    text = subject.strip()

    m = _SPLIT.search(text)
    if m:
        old, new = float(m.group(1)), float(m.group(2))
        if new > 0 and old > new:
            # Face value 10 -> 2 means each share becomes 5.
            return {"action_type": "split", "ratio_from": new, "ratio_to": old, "amount": 0.0}

    m = _BONUS.search(text)
    if m:
        new, held = float(m.group(1)), float(m.group(2))
        if held > 0:
            return {"action_type": "bonus", "ratio_from": held, "ratio_to": new, "amount": 0.0}

    if _DIVIDEND.search(text):
        amt = _AMOUNT.search(text)
        return {
            "action_type": "dividend",
            "ratio_from": 1.0,
            "ratio_to": 1.0,
            "amount": float(amt.group(1)) if amt else 0.0,
        }
    return None


def normalize_corporate_actions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Structured actions plus the rows we could not parse.

    Unparsed rows are returned rather than dropped so they can be reviewed -
    a silently missed bonus issue looks exactly like a 50% crash.
    """
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    parsed, unparsed = [], []
    for _, row in raw.iterrows():
        subject = row.get("subject", "")
        action = parse_corporate_action(subject)
        ex_date = pd.to_datetime(row.get("exDate"), format="%d-%b-%Y", errors="coerce")
        if action is None or pd.isna(ex_date):
            unparsed.append({"isin": row.get("isin"), "symbol": row.get("symbol"),
                             "ex_date": row.get("exDate"), "subject": subject})
            continue
        parsed.append({
            "isin": row.get("isin"), "symbol": row.get("symbol"),
            "ex_date": ex_date, **action, "subject": subject,
        })

    if unparsed:
        log.info("%d corporate actions could not be parsed (kept for review)", len(unparsed))
    return pd.DataFrame(parsed), pd.DataFrame(unparsed)


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def bhavcopy_to_panels(bhav: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long bhavcopy rows -> wide (date x isin) panels."""
    if bhav.empty:
        return {}

    df = bhav.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "isin"], keep="last")

    panels = {}
    for field in ("close", "open", "high", "low", "volume", "traded_value"):
        if field in df.columns:
            panel = df.pivot(index="date", columns="isin", values=field).sort_index()
            panel.index.name = "date"
            panels[field] = panel
    return panels


def build_symbol_history(bhav: pd.DataFrame) -> pd.DataFrame:
    """Derive symbol-change history by observing ISIN/symbol pairs over time.

    NSE does not publish a tidy rename log, but the bhavcopy archive shows
    which symbol an ISIN traded under on every date, so the history can be
    reconstructed from the data itself.
    """
    if bhav.empty:
        return pd.DataFrame(columns=["isin", "symbol", "valid_from", "valid_to"])

    df = bhav[["date", "isin", "symbol"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    grouped = (
        df.groupby(["isin", "symbol"])["date"]
        .agg(valid_from="min", valid_to="max")
        .reset_index()
        .sort_values(["isin", "valid_from"])
    )
    # An ISIN's most recent symbol is still current.
    last = grouped.groupby("isin")["valid_to"].transform("max")
    grouped.loc[grouped["valid_to"] == last, "valid_to"] = pd.NaT
    return grouped


def infer_delistings(bhav: pd.DataFrame, as_of: date | None = None,
                     absent_days: int = 30) -> pd.DataFrame:
    """ISINs that stopped appearing in the archive - i.e. delisted or suspended."""
    if bhav.empty:
        return pd.DataFrame(columns=["isin", "last_seen"])

    df = bhav[["date", "isin"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    last_seen = df.groupby("isin")["date"].max()
    cutoff = pd.Timestamp(as_of or df["date"].max()) - pd.Timedelta(days=absent_days)

    gone = last_seen[last_seen < cutoff]
    return pd.DataFrame({"isin": gone.index, "last_seen": gone.values})
