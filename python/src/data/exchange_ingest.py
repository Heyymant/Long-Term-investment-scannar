"""Build the framework's data layer directly from NSE/BSE public data.

This is the glue that turns raw exchange downloads into the structures the
rest of the system expects, and it fixes two long-standing caveats:

* **Survivorship bias.** Prices come from the bhavcopy archive, which records
  every security that traded each day. Companies that later delisted stay in
  the history where they belong, instead of vanishing as they do from a live
  broker instrument dump.
* **Look-ahead in fundamentals.** Filings are stamped with the exchange
  dissemination time, so a number becomes usable at the moment the market
  actually received it - not at an assumed lag after quarter end.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, get_logger
from .bse_client import BSEClient
from .corporate_actions import ActionType, CorporateActions
from .nse_client import NSEClient, build_price_panel
from .security_master import SecurityMaster
from .universe import IndexMembership

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Security master
# --------------------------------------------------------------------------- #
def build_security_master(
    cfg: AppConfig,
    nse: NSEClient | None = None,
    bse: BSEClient | None = None,
    index: str = "NIFTY500",
) -> SecurityMaster:
    """Assemble an ISIN-keyed master from NSE's equity list, enriched with
    index membership (for sectors) and BSE scrip codes (for cross-referencing).
    """
    nse = nse or NSEClient(cache_dir=cfg.paths.data_dir / "nse")
    equities = nse.equity_list()
    if equities.empty:
        raise RuntimeError("NSE equity list was empty")

    master = pd.DataFrame({
        "isin": equities["isin"].astype(str).str.strip(),
        "symbol": equities["symbol"].astype(str).str.strip(),
        "name": equities.get("name", equities["symbol"]).astype(str).str.strip(),
        "kite_token": pd.NA,
        "exchange": "NSE",
        "sector": pd.NA,
        "industry": pd.NA,
        "listing_date": equities.get("listing_date", pd.NaT),
        "delisting_date": pd.NaT,
        "fundamentals_id": equities["symbol"].astype(str).str.strip(),
        "status": "active",
    })
    master = master[master["isin"].str.startswith("INE", na=False)]
    master = master.drop_duplicates(subset=["isin"]).reset_index(drop=True)

    # NSE's index CSVs carry the official industry classification.
    sectors = _collect_sectors(nse)
    if not sectors.empty:
        master = master.merge(sectors, on="isin", how="left", suffixes=("", "_idx"))
        master["sector"] = master["sector_idx"].combine_first(master["sector"])
        master["industry"] = master["sector"]
        master = master.drop(columns=[c for c in master.columns if c.endswith("_idx")])

    # BSE scrip codes let us reconcile the two feeds later.
    if bse is not None:
        try:
            scrips = bse.scrip_master()
            if not scrips.empty and "isin" in scrips.columns:
                codes = scrips[["isin", "scrip_code"]].dropna().drop_duplicates("isin")
                master = master.merge(codes, on="isin", how="left")
                log.info("Matched %d BSE scrip codes", master["scrip_code"].notna().sum())
        except Exception as exc:  # noqa: BLE001 - BSE is optional enrichment
            log.warning("Could not attach BSE scrip codes: %s", exc)

    history = master[["isin", "symbol"]].copy()
    history["valid_from"] = master["listing_date"].fillna(pd.Timestamp("1990-01-01"))
    history["valid_to"] = pd.NaT

    log.info(
        "Security master: %d securities, %d with a sector",
        len(master), int(master["sector"].notna().sum()),
    )
    return SecurityMaster(
        securities=master,
        symbol_history=history,
        mergers=pd.DataFrame(columns=["from_isin", "to_isin", "effective_date", "ratio", "note"]),
    )


def _collect_sectors(nse: NSEClient) -> pd.DataFrame:
    """Union index constituent files to maximize sector coverage."""
    frames = []
    for index in ("NIFTY500", "NIFTYMIDCAP150", "NIFTYSMALLCAP250"):
        try:
            df = nse.index_constituents(index)
            if not df.empty and {"isin", "sector"}.issubset(df.columns):
                frames.append(df[["isin", "sector"]])
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not load %s constituents: %s", index, exc)
    if not frames:
        return pd.DataFrame(columns=["isin", "sector"])
    out = pd.concat(frames, ignore_index=True).dropna(subset=["isin"])
    return out.drop_duplicates(subset=["isin"]).rename(columns={"sector": "sector_idx"})


# --------------------------------------------------------------------------- #
# Prices (survivorship-safe)
# --------------------------------------------------------------------------- #
def build_price_history(
    cfg: AppConfig,
    start: str | date,
    end: str | date,
    nse: NSEClient | None = None,
    key: str = "isin",
) -> dict[str, pd.DataFrame]:
    """Download bhavcopies and pivot into price/volume panels.

    Keyed on ISIN rather than symbol, so a company that renames mid-history
    stays a single continuous series instead of splitting into two.
    """
    nse = nse or NSEClient(cache_dir=cfg.paths.data_dir / "nse")
    bhav = nse.bhavcopy_range(start, end)
    if bhav.empty:
        raise RuntimeError(f"no bhavcopy data between {start} and {end}")

    if key == "isin" and "isin" not in bhav.columns:
        log.warning("bhavcopy has no ISIN column; falling back to symbol keys")
        key = "symbol"
    if key == "isin":
        bhav = bhav[bhav["isin"].astype(str).str.startswith("INE")]

    panels = {
        "close": build_price_panel(bhav, "close", key),
        "open": build_price_panel(bhav, "open", key),
        "high": build_price_panel(bhav, "high", key),
        "low": build_price_panel(bhav, "low", key),
        "volume": build_price_panel(bhav, "volume", key),
        "turnover": build_price_panel(bhav, "turnover", key),
    }

    close = panels["close"]
    # A name that stops appearing was delisted or suspended - that is signal,
    # not missing data, and it is exactly what survivorship bias hides.
    last_seen = close.apply(lambda col: col.last_valid_index())
    delisted = last_seen[last_seen < close.index[-1] - pd.Timedelta(days=30)]
    log.info(
        "Price history: %d securities, %s..%s (%d stopped trading before the end - "
        "these are the names a survivorship-biased universe would silently drop)",
        close.shape[1], close.index[0].date(), close.index[-1].date(), len(delisted),
    )
    return panels


def save_price_panels(panels: dict[str, pd.DataFrame], cfg: AppConfig) -> Path:
    d = cfg.paths.data_dir / "panels"
    d.mkdir(parents=True, exist_ok=True)
    for name, df in panels.items():
        if df is not None and not df.empty:
            df.to_parquet(d / f"{name}.parquet")
    return d


def load_price_panels(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    d = cfg.paths.data_dir / "panels"
    out: dict[str, pd.DataFrame] = {}
    if not d.exists():
        return out
    for path in d.glob("*.parquet"):
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        out[path.stem] = df.sort_index()
    return out


def infer_delistings(close: pd.DataFrame, grace_days: int = 30) -> pd.DataFrame:
    """Infer delisting dates from when a security stops appearing."""
    if close.empty:
        return pd.DataFrame(columns=["isin", "delisting_date"])
    cutoff = close.index[-1] - pd.Timedelta(days=grace_days)
    last_seen = close.apply(lambda col: col.last_valid_index())
    gone = last_seen[last_seen.notna() & (last_seen < cutoff)]
    return pd.DataFrame({"isin": gone.index, "delisting_date": gone.values})


# --------------------------------------------------------------------------- #
# Corporate actions
# --------------------------------------------------------------------------- #
def build_corporate_actions(
    cfg: AppConfig,
    start: str | date,
    end: str | date,
    nse: NSEClient | None = None,
    symbol_to_isin: dict[str, str] | None = None,
) -> CorporateActions:
    """Parse NSE corporate-action notices into split/bonus/dividend records."""
    nse = nse or NSEClient(cache_dir=cfg.paths.data_dir / "nse", allow_api=True)

    frames = []
    cursor, stop = pd.Timestamp(start), pd.Timestamp(end)
    while cursor <= stop:
        chunk_end = min(cursor + pd.DateOffset(months=6) - pd.Timedelta(days=1), stop)
        try:
            df = nse.corporate_actions(cursor, chunk_end)
            if not df.empty:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            log.warning("corporate actions %s..%s failed: %s",
                        cursor.date(), chunk_end.date(), exc)
        cursor = chunk_end + pd.Timedelta(days=1)

    if not frames:
        log.warning("No corporate actions retrieved; prices will be unadjusted")
        return CorporateActions.empty()

    raw = pd.concat(frames, ignore_index=True)
    records = []
    for row in raw.itertuples(index=False):
        subject = str(getattr(row, "subject", "") or getattr(row, "purpose", ""))
        ex_date = getattr(row, "ex_date", None)
        if pd.isna(ex_date):
            continue

        isin = getattr(row, "isin", None)
        if not isin and symbol_to_isin:
            isin = symbol_to_isin.get(str(getattr(row, "symbol", "")))
        if not isin:
            continue

        parsed = _parse_action(subject)
        if parsed:
            records.append({"isin": isin, "ex_date": ex_date, **parsed})

    if not records:
        return CorporateActions.empty()

    actions = pd.DataFrame(records)
    log.info(
        "Corporate actions: %d parsed (%s)",
        len(actions), actions["action_type"].value_counts().to_dict(),
    )
    return CorporateActions(actions)


def _parse_action(subject: str) -> dict | None:
    """Extract structured ratios from NSE's free-text action descriptions.

    Examples seen in the feed:
      "Face Value Split From Rs 10/- To Rs 2/-"   -> 1:5 split
      "Bonus 1:2"                                  -> 1 new share per 2 held
      "Dividend - Rs 12 Per Share"                 -> Rs 12 dividend
    """
    import re

    text = subject.strip()
    lower = text.lower()

    if "split" in lower:
        nums = re.findall(r"(?:rs\.?\s*)?(\d+(?:\.\d+)?)", lower)
        if len(nums) >= 2:
            old, new = float(nums[0]), float(nums[1])
            if new > 0 and old > new:
                return {"action_type": ActionType.SPLIT.value,
                        "ratio_from": new, "ratio_to": old, "amount": 0.0}
        return None

    if "bonus" in lower:
        match = re.search(r"(\d+)\s*[:/]\s*(\d+)", lower)
        if match:
            new, held = float(match.group(1)), float(match.group(2))
            if held > 0:
                return {"action_type": ActionType.BONUS.value,
                        "ratio_from": held, "ratio_to": new, "amount": 0.0}
        return None

    if "dividend" in lower:
        match = re.search(r"(?:rs\.?|inr)\s*(\d+(?:\.\d+)?)", lower)
        if match:
            return {"action_type": ActionType.DIVIDEND.value,
                    "ratio_from": 1.0, "ratio_to": 1.0, "amount": float(match.group(1))}
        return None

    return None


# --------------------------------------------------------------------------- #
# Point-in-time index membership
# --------------------------------------------------------------------------- #
def build_index_membership(
    cfg: AppConfig,
    index: str = "NIFTY500",
    nse: NSEClient | None = None,
    close: pd.DataFrame | None = None,
) -> IndexMembership:
    """Approximate point-in-time membership.

    NSE does not publish a machine-readable history of index changes, so this
    combines today's constituents with each security's observed trading window
    from the bhavcopy archive. That is weaker than a true change log, but far
    better than assuming today's members existed and were investable
    throughout - which is the classic survivorship mistake.
    """
    nse = nse or NSEClient(cache_dir=cfg.paths.data_dir / "nse")
    members = nse.index_constituents(index)
    if members.empty:
        return IndexMembership.empty()

    rows = []
    for isin in members["isin"].dropna().unique():
        start = pd.Timestamp("1990-01-01")
        end = pd.NaT
        if close is not None and isin in close.columns:
            series = close[isin].dropna()
            if not series.empty:
                start = series.index[0]
                # Treat a long gap before the panel ends as a delisting.
                if series.index[-1] < close.index[-1] - pd.Timedelta(days=30):
                    end = series.index[-1]
        rows.append({
            "isin": isin, "index_name": index.upper().replace(" ", ""),
            "start_date": start, "end_date": end,
        })

    log.info("Index membership for %s: %d securities", index, len(rows))
    log.warning(
        "Membership start dates are inferred from first trade, not from index "
        "inclusion notices. Supply data/index_membership.csv for exact history."
    )
    return IndexMembership(pd.DataFrame(rows))
