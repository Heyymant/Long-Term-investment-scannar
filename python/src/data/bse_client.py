"""BSE India public data client.

BSE is the secondary source. It matters for two reasons:

* **Cross-verification.** When NSE and BSE disagree on an announcement date or
  a corporate action ratio, that disagreement is itself a data-quality signal
  worth surfacing rather than silently picking one.
* **Coverage.** Some smaller companies list only on BSE, and BSE's forward
  results calendar is often populated earlier than NSE's event calendar.

Unlike NSE, BSE's JSON API accepts ordinary HTTP clients - it only requires a
browser User-Agent and a `Referer` of https://www.bseindia.com/. No TLS
impersonation is needed, so this path carries no terms-of-service ambiguity.
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config import get_logger

log = get_logger(__name__)

API = "https://api.bseindia.com/BseIndiaAPI/api"
SITE = "https://www.bseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{SITE}/",
    "Origin": SITE,
}


class BSEError(RuntimeError):
    pass


@dataclass
class BSEClient:
    cache_dir: Path | None = None
    rate_limit_per_sec: float = 1.5
    timeout: int = 30
    max_retries: int = 3
    session: requests.Session = field(default_factory=requests.Session, repr=False)
    _last_call: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.session.headers.update(HEADERS)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        if self.rate_limit_per_sec <= 0:
            return
        gap = 1.0 / self.rate_limit_per_sec
        elapsed = time.monotonic() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{API}/{path}"
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                if not resp.text.strip():
                    return None
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                wait = 2**attempt
                log.debug("BSE request failed (%s), retry in %ds: %s", path, wait, exc)
                time.sleep(wait)
        log.warning("BSE request gave up after %d attempts: %s", self.max_retries, path)
        return None

    # -- corporate filings -----------------------------------------------------
    def announcements(
        self, from_date: date | str, to_date: date | str,
        scrip_code: str = "", category: str = "-1", max_pages: int = 50,
    ) -> pd.DataFrame:
        """Corporate announcements, paginated until the API returns nothing.

        BSE answers 200 for any page number, so an empty `Table` is the only
        reliable end-of-data signal.
        """
        rows: list[dict] = []
        for page in range(1, max_pages + 1):
            payload = self._get("AnnSubCategoryGetData/w", {
                "pageno": page,
                "strCat": category,
                "strPrevDate": _yyyymmdd(from_date),
                "strToDate": _yyyymmdd(to_date),
                "strScrip": scrip_code,
                "strSearch": "P",
                "strType": "C",
                "subcategory": "",
            })
            table = _table(payload)
            if not table:
                break
            rows.extend(table)

        if not rows:
            # BSE reshuffles this endpoint's parameters periodically and returns
            # 200-with-empty rather than an error. NSE covers announcements, so
            # this is a soft failure, not a blocker.
            log.info(
                "BSE announcements returned nothing for %s..%s. Use the NSE "
                "client for announcements; BSE's parameters may have changed.",
                from_date, to_date,
            )
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("News_submission_dt", "DissemDT", "News_Dt"):
            if col in df.columns:
                df["announce_datetime"] = pd.to_datetime(df[col], errors="coerce")
                break
        return df

    def result_calendar(
        self, from_date: date | str, to_date: date | str,
    ) -> pd.DataFrame:
        """Forthcoming board meetings for results - the forward earnings calendar."""
        rows: list[dict] = []
        for path, params in (
            ("BoardMeeting_New/w", {
                "pageno": 1, "strCat": "Board Meeting",
                "strPrevDate": _yyyymmdd(from_date), "strToDate": _yyyymmdd(to_date),
                "strScrip": "", "strSearch": "P", "strType": "C",
            }),
            ("BoardMeeting/w", {
                "pageno": 1, "strCat": "-1",
                "strPrevDate": _yyyymmdd(from_date), "strToDate": _yyyymmdd(to_date),
                "strScrip": "", "strSearch": "P", "strType": "C",
            }),
        ):
            rows = _table(self._get(path, params))
            if rows:
                break

        if not rows:
            log.info(
                "BSE result calendar unavailable for %s..%s. Use NSE's "
                "event_calendar() instead - it covers the same board meetings.",
                from_date, to_date,
            )
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("Meeting_Dt", "MeetingDate", "BoardMeetingDate"):
            if col in df.columns:
                df["event_date"] = pd.to_datetime(df[col], errors="coerce")
                break
        if "Purpose" in df.columns:
            df["is_results"] = df["Purpose"].astype(str).str.contains(
                "result|financial", case=False, na=False
            )
        return df

    def corporate_actions(
        self, from_date: date | str, to_date: date | str, scrip_code: str = "",
    ) -> pd.DataFrame:
        payload = self._get("DefaultData/w", {
            "ddlcategorys": "E",
            "ddlindustrys": "",
            "Fdate": _yyyymmdd(from_date),
            "TDate": _yyyymmdd(to_date),
            "scripcode": scrip_code,
            "segment": "0",
            "strSearch": "S",
        })
        rows = _table(payload)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in ("Ex_date", "EX_DATE", "ExDate"):
            if col in df.columns:
                df["ex_date"] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                break
        return df

    # -- market data -----------------------------------------------------------
    def bhavcopy(self, on: date | str, use_cache: bool = True) -> pd.DataFrame:
        """BSE end-of-day snapshot (includes securities not listed on NSE)."""
        day = pd.Timestamp(on).date()
        cached = None
        if self.cache_dir:
            d = self.cache_dir / "bse_bhavcopy"
            d.mkdir(parents=True, exist_ok=True)
            cached = d / f"{day:%Y%m%d}.parquet"
            if use_cache and cached.exists():
                return pd.read_parquet(cached)

        df = self._download_bhavcopy(day)
        out = _normalize_bse_bhavcopy(df, day) if df is not None else pd.DataFrame()
        if cached:
            out.to_parquet(cached, index=False)
        return out

    def _download_bhavcopy(self, day: date) -> pd.DataFrame | None:
        candidates = [
            f"{SITE}/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{day:%Y%m%d}_F_0000.CSV",
            f"{SITE}/download/BhavCopy/Equity/EQ{day:%d%m%y}_CSV.ZIP",
        ]
        for url in candidates:
            try:
                self._throttle()
                resp = self.session.get(
                    url, headers={**HEADERS, "Accept": "*/*"}, timeout=self.timeout
                )
                ctype = resp.headers.get("content-type", "")
                # BSE serves an HTML error page rather than a 404.
                if resp.status_code != 200 or "html" in ctype.lower():
                    continue
                if resp.content[:2] == b"PK":
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                        if not names:
                            continue
                        with zf.open(names[0]) as fh:
                            return pd.read_csv(fh)
                return pd.read_csv(io.BytesIO(resp.content))
            except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
                log.debug("BSE bhavcopy attempt failed (%s): %s", url, exc)
        return None

    def scrip_master(self) -> pd.DataFrame:
        """BSE scrip codes with ISIN - needed to join BSE data to our master."""
        payload = self._get("ListofScripData/w", {
            "Group": "", "Scripcode": "", "industry": "",
            "segment": "Equity", "status": "Active",
        })
        rows = payload if isinstance(payload, list) else _table(payload)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {
            "SCRIP_CD": "scrip_code", "Scrip_Code": "scrip_code",
            "ISIN_NUMBER": "isin", "ISIN_NO": "isin",
            "scrip_id": "symbol", "Scrip_Name": "name", "SCRIP_NAME": "name",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ("isin", "symbol"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        return df


# --------------------------------------------------------------------------- #
def _table(payload: Any) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("Table", "table", "Data", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def _yyyymmdd(d: date | str | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _normalize_bse_bhavcopy(df: pd.DataFrame, day: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={
        "TCKRSYMB": "SC_NAME", "FININSTRMID": "SC_CODE", "CLSPRIC": "CLOSE",
        "OPNPRIC": "OPEN", "HGHPRIC": "HIGH", "LWPRIC": "LOW",
        "TTLTRADGVOL": "NO_OF_SHRS", "TTLTRFVAL": "NET_TURNOV", "ISIN": "ISIN_CODE",
    })
    if "SC_CODE" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date": pd.Timestamp(day),
        "scrip_code": df["SC_CODE"].astype(str).str.strip(),
        "name": df["SC_NAME"].astype(str).str.strip() if "SC_NAME" in df.columns else None,
    })
    for src, dst in (("OPEN", "open"), ("HIGH", "high"), ("LOW", "low"),
                     ("CLOSE", "close"), ("NO_OF_SHRS", "volume"), ("NET_TURNOV", "turnover")):
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else pd.NA
    if "ISIN_CODE" in df.columns:
        out["isin"] = df["ISIN_CODE"].astype(str).str.strip()
    return out.dropna(subset=["close"]).reset_index(drop=True)


def cross_verify_announcements(
    nse: pd.DataFrame, bse: pd.DataFrame, tolerance_hours: int = 24,
) -> pd.DataFrame:
    """Compare NSE and BSE announcement timestamps for the same filings.

    Large disagreements usually mean one exchange's feed is stale. Surfacing
    that is more useful than silently trusting whichever source we happened to
    query first.
    """
    if nse.empty or bse.empty:
        return pd.DataFrame()

    n = nse[["isin", "announce_datetime"]].dropna().copy() if "isin" in nse.columns else pd.DataFrame()
    b = bse[["isin", "announce_datetime"]].dropna().copy() if "isin" in bse.columns else pd.DataFrame()
    if n.empty or b.empty:
        return pd.DataFrame()

    merged = n.merge(b, on="isin", how="inner", suffixes=("_nse", "_bse"))
    merged["delta_hours"] = (
        merged["announce_datetime_nse"] - merged["announce_datetime_bse"]
    ).dt.total_seconds() / 3600.0
    merged["disagrees"] = merged["delta_hours"].abs() > tolerance_hours
    return merged
