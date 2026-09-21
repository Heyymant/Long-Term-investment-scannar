"""ISIN-keyed security master.

The canonical identity table. Everything else (prices, fundamentals, index
constituents, earnings) joins through here, because NSE symbols are *not*
stable over time: companies rename, merge, and delist. Keying on ISIN and
tracking symbol history is what keeps a long backtest from silently mixing up
two different companies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

SECURITY_COLUMNS = [
    "isin",              # canonical key
    "symbol",            # current NSE trading symbol
    "name",
    "kite_token",        # Kite instrument_token
    "exchange",
    "sector",
    "industry",
    "listing_date",
    "delisting_date",    # NaT if still listed
    "fundamentals_id",   # provider-specific id
    "status",            # active | delisted | merged
]

SYMBOL_HISTORY_COLUMNS = ["isin", "symbol", "valid_from", "valid_to"]
MERGER_COLUMNS = ["from_isin", "to_isin", "effective_date", "ratio", "note"]


@dataclass
class SecurityMaster:
    """Identity resolution across time.

    securities:      one row per ISIN (current state)
    symbol_history:  (isin, symbol, valid_from, valid_to) intervals
    mergers:         corporate identity changes (from_isin -> to_isin)
    """

    securities: pd.DataFrame
    symbol_history: pd.DataFrame
    mergers: pd.DataFrame

    # -- construction ----------------------------------------------------------
    @classmethod
    def empty(cls) -> SecurityMaster:
        return cls(
            securities=pd.DataFrame(columns=SECURITY_COLUMNS),
            symbol_history=pd.DataFrame(columns=SYMBOL_HISTORY_COLUMNS),
            mergers=pd.DataFrame(columns=MERGER_COLUMNS),
        )

    @classmethod
    def from_kite_instruments(
        cls,
        instruments: pd.DataFrame,
        sectors: pd.DataFrame | None = None,
    ) -> SecurityMaster:
        """Build from a Kite instruments dump (NSE equity segment).

        Kite's dump has no ISIN, so when it is absent we fall back to the
        symbol as a placeholder key and flag it - callers should enrich with a
        real ISIN source when available.
        """
        df = instruments.copy()
        if "exchange" in df.columns:
            df = df[df["exchange"] == "NSE"]
        if "instrument_type" in df.columns:
            df = df[df["instrument_type"] == "EQ"]
        if "segment" in df.columns:
            df = df[df["segment"].isin(["NSE", "NSE-EQ"])]

        out = pd.DataFrame({
            "isin": df["isin"] if "isin" in df.columns else "SYN:" + df["tradingsymbol"].astype(str),
            "symbol": df["tradingsymbol"].astype(str),
            "name": df.get("name", df["tradingsymbol"]).astype(str),
            "kite_token": pd.to_numeric(df.get("instrument_token"), errors="coerce").astype("Int64"),
            "exchange": "NSE",
            "sector": pd.NA,
            "industry": pd.NA,
            "listing_date": pd.NaT,
            "delisting_date": pd.NaT,
            "fundamentals_id": pd.NA,
            "status": "active",
        })
        out["isin"] = out["isin"].fillna("SYN:" + out["symbol"])
        out = out.drop_duplicates(subset=["isin"]).reset_index(drop=True)

        if sectors is not None and not sectors.empty:
            out = _attach_sectors(out, sectors)

        hist = out[["isin", "symbol"]].copy()
        hist["valid_from"] = pd.Timestamp("1990-01-01")
        hist["valid_to"] = pd.NaT

        log.info("Security master built: %d instruments", len(out))
        return cls(securities=out, symbol_history=hist, mergers=pd.DataFrame(columns=MERGER_COLUMNS))

    # -- resolution ------------------------------------------------------------
    def symbol_at(self, isin: str, on: date | pd.Timestamp) -> str | None:
        """Which trading symbol did this ISIN use on a given date?"""
        ts = pd.Timestamp(on)
        h = self.symbol_history
        rows = h[h["isin"] == isin]
        if rows.empty:
            cur = self.securities.loc[self.securities["isin"] == isin, "symbol"]
            return None if cur.empty else str(cur.iloc[0])
        mask = (rows["valid_from"] <= ts) & (rows["valid_to"].isna() | (rows["valid_to"] >= ts))
        hit = rows[mask]
        if hit.empty:
            return None
        return str(hit.iloc[-1]["symbol"])

    def resolve_symbol(self, symbol: str, on: date | pd.Timestamp | None = None) -> str | None:
        """Map a (possibly historical) symbol back to its ISIN."""
        h = self.symbol_history
        rows = h[h["symbol"] == symbol]
        if rows.empty:
            cur = self.securities.loc[self.securities["symbol"] == symbol, "isin"]
            return None if cur.empty else str(cur.iloc[0])
        if on is not None:
            ts = pd.Timestamp(on)
            mask = (rows["valid_from"] <= ts) & (rows["valid_to"].isna() | (rows["valid_to"] >= ts))
            rows = rows[mask] if mask.any() else rows
        return str(rows.iloc[-1]["isin"])

    def follow_mergers(self, isin: str, on: date | pd.Timestamp | None = None) -> str:
        """Follow a merger chain to the surviving entity."""
        if self.mergers.empty:
            return isin
        ts = pd.Timestamp(on) if on is not None else None
        current, seen = isin, {isin}
        while True:
            m = self.mergers[self.mergers["from_isin"] == current]
            if ts is not None and not m.empty:
                m = m[pd.to_datetime(m["effective_date"]) <= ts]
            if m.empty:
                return current
            nxt = str(m.iloc[-1]["to_isin"])
            if nxt in seen:  # guard against cyclic data
                log.warning("Cyclic merger chain at %s; stopping", nxt)
                return current
            seen.add(nxt)
            current = nxt

    def token_for(self, isin: str) -> int | None:
        row = self.securities.loc[self.securities["isin"] == isin, "kite_token"]
        if row.empty or pd.isna(row.iloc[0]):
            return None
        return int(row.iloc[0])

    def active_on(self, on: date | pd.Timestamp) -> pd.DataFrame:
        """Securities tradeable on a date (listed, not yet delisted)."""
        ts = pd.Timestamp(on)
        s = self.securities
        listed = s["listing_date"].isna() | (pd.to_datetime(s["listing_date"]) <= ts)
        alive = s["delisting_date"].isna() | (pd.to_datetime(s["delisting_date"]) > ts)
        return s[listed & alive]

    def sector_map(self) -> dict[str, str]:
        s = self.securities
        return dict(zip(s["isin"], s["sector"].fillna("UNKNOWN")))

    def symbol_to_isin_map(self) -> dict[str, str]:
        return dict(zip(self.securities["symbol"], self.securities["isin"]))

    # -- persistence -----------------------------------------------------------
    def save(self, data_dir: str | Path) -> None:
        d = Path(data_dir) / "security_master"
        d.mkdir(parents=True, exist_ok=True)
        self.securities.to_parquet(d / "securities.parquet", index=False)
        self.symbol_history.to_parquet(d / "symbol_history.parquet", index=False)
        self.mergers.to_parquet(d / "mergers.parquet", index=False)
        log.info("Security master saved to %s", d)

    @classmethod
    def load(cls, data_dir: str | Path) -> SecurityMaster:
        d = Path(data_dir) / "security_master"
        if not (d / "securities.parquet").exists():
            return cls.empty()

        def _read(name: str, cols: list[str]) -> pd.DataFrame:
            p = d / name
            return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=cols)

        return cls(
            securities=_read("securities.parquet", SECURITY_COLUMNS),
            symbol_history=_read("symbol_history.parquet", SYMBOL_HISTORY_COLUMNS),
            mergers=_read("mergers.parquet", MERGER_COLUMNS),
        )

    # -- mutation --------------------------------------------------------------
    def record_symbol_change(
        self, isin: str, old_symbol: str, new_symbol: str, effective: date | pd.Timestamp
    ) -> None:
        ts = pd.Timestamp(effective)
        h = self.symbol_history
        mask = (h["isin"] == isin) & (h["symbol"] == old_symbol) & h["valid_to"].isna()
        h.loc[mask, "valid_to"] = ts - pd.Timedelta(days=1)
        self.symbol_history = pd.concat(
            [h, pd.DataFrame([{"isin": isin, "symbol": new_symbol, "valid_from": ts, "valid_to": pd.NaT}])],
            ignore_index=True,
        )
        self.securities.loc[self.securities["isin"] == isin, "symbol"] = new_symbol
        log.info("Symbol change %s: %s -> %s on %s", isin, old_symbol, new_symbol, ts.date())

    def record_delisting(self, isin: str, effective: date | pd.Timestamp, status: str = "delisted") -> None:
        self.securities.loc[self.securities["isin"] == isin, ["delisting_date", "status"]] = [
            pd.Timestamp(effective), status
        ]

    def record_merger(
        self, from_isin: str, to_isin: str, effective: date | pd.Timestamp,
        ratio: float = 1.0, note: str = "",
    ) -> None:
        self.mergers = pd.concat(
            [self.mergers, pd.DataFrame([{
                "from_isin": from_isin, "to_isin": to_isin,
                "effective_date": pd.Timestamp(effective), "ratio": ratio, "note": note,
            }])],
            ignore_index=True,
        )
        self.record_delisting(from_isin, effective, status="merged")


def build_from_nse(
    equity_list: pd.DataFrame,
    constituents: pd.DataFrame | None = None,
    kite_instruments: pd.DataFrame | None = None,
    symbol_history: pd.DataFrame | None = None,
    delistings: pd.DataFrame | None = None,
) -> SecurityMaster:
    """Build the security master from NSE's own published data.

    This is strictly better than deriving it from the Kite instrument dump,
    which carries no ISIN: NSE's EQUITY_L.csv gives real ISINs and listing
    dates, and the index files supply the industry classification used for
    sector neutralization.
    """
    if equity_list.empty:
        log.warning("Empty NSE equity list; cannot build a security master")
        return SecurityMaster.empty()

    out = pd.DataFrame({
        "isin": equity_list["isin"].astype(str).str.strip(),
        "symbol": equity_list["symbol"].astype(str).str.strip(),
        "name": equity_list.get("name", equity_list["symbol"]).astype(str),
        "kite_token": pd.Series([pd.NA] * len(equity_list), dtype="Int64"),
        "exchange": "NSE",
        "sector": pd.NA,
        "industry": pd.NA,
        "listing_date": pd.to_datetime(equity_list.get("listing_date"), errors="coerce"),
        "delisting_date": pd.NaT,
        "fundamentals_id": equity_list["symbol"].astype(str),
        "status": "active",
    }).drop_duplicates(subset=["isin"]).reset_index(drop=True)

    # Sector/industry from the index constituent files.
    if constituents is not None and not constituents.empty:
        sector_map = (
            constituents.dropna(subset=["isin"])
            .drop_duplicates(subset=["isin"])
            .set_index("isin")["sector"]
        )
        out["sector"] = out["isin"].map(sector_map)
        out["industry"] = out["sector"]
        log.info("Sector classification attached for %d names", int(out["sector"].notna().sum()))

    # Kite tokens (for live quotes) joined on symbol.
    if kite_instruments is not None and not kite_instruments.empty:
        token_col = "instrument_token" if "instrument_token" in kite_instruments.columns else None
        sym_col = "tradingsymbol" if "tradingsymbol" in kite_instruments.columns else None
        if token_col and sym_col:
            tokens = (
                kite_instruments.drop_duplicates(subset=[sym_col])
                .set_index(sym_col)[token_col]
            )
            out["kite_token"] = pd.to_numeric(out["symbol"].map(tokens), errors="coerce").astype("Int64")
            log.info("Kite tokens matched for %d names", int(out["kite_token"].notna().sum()))

    # Symbol history reconstructed from the bhavcopy archive.
    if symbol_history is not None and not symbol_history.empty:
        hist = symbol_history[["isin", "symbol", "valid_from", "valid_to"]].copy()
    else:
        hist = out[["isin", "symbol"]].copy()
        hist["valid_from"] = pd.Timestamp("1990-01-01")
        hist["valid_to"] = pd.NaT

    master = SecurityMaster(
        securities=out, symbol_history=hist, mergers=pd.DataFrame(columns=MERGER_COLUMNS)
    )

    if delistings is not None and not delistings.empty:
        for _, row in delistings.iterrows():
            master.record_delisting(row["isin"], row["last_seen"])
        log.info("Marked %d securities as delisted/suspended", len(delistings))

    log.info("Security master from NSE: %d instruments with real ISINs", len(out))
    return master


def _attach_sectors(securities: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """Join sector/industry by ISIN when available, else by symbol."""
    key = "isin" if "isin" in sectors.columns else "symbol"
    cols = [c for c in ("sector", "industry") if c in sectors.columns]
    if not cols:
        return securities
    merged = securities.merge(sectors[[key, *cols]], on=key, how="left", suffixes=("", "_new"))
    for c in cols:
        new = f"{c}_new"
        if new in merged.columns:
            merged[c] = merged[new].combine_first(merged[c])
            merged = merged.drop(columns=[new])
    return merged
