"""Zerodha Kite Connect data access - READ ONLY.

This module never places, modifies or cancels orders. It only reads:
instruments, historical candles, holdings, positions, quotes and the
orders/trades book (for the execution journal).

Kite constraints handled here:
  * day candles are capped at 2000 days per request -> we paginate
  * NSE day data generally reaches ~2005/06 (instrument-dependent)
  * rate limits -> throttled, with retry/backoff
  * access tokens expire daily -> interactive login flow
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import AppConfig, env, get_logger

log = get_logger(__name__)

MAX_DAYS_PER_REQUEST = 2000
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class KiteAuthError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Session / auth
# --------------------------------------------------------------------------- #
@dataclass
class KiteSession:
    """Wraps kiteconnect with a cached daily access token."""

    api_key: str
    api_secret: str
    access_token: str | None = None
    token_file: Path | None = None
    _kite: Any = None

    @classmethod
    def from_env(cls, data_dir: str | Path | None = None) -> KiteSession:
        api_key = env("KITE_API_KEY")
        api_secret = env("KITE_API_SECRET")
        if not api_key or not api_secret:
            raise KiteAuthError(
                "KITE_API_KEY / KITE_API_SECRET not set. Copy .env.example to .env and fill them in."
            )
        token_file = Path(data_dir) / "kite_token.json" if data_dir else None
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            access_token=env("KITE_ACCESS_TOKEN") or None,
            token_file=token_file,
        )

    def login_url(self) -> str:
        return f"https://kite.zerodha.com/connect/login?v=3&api_key={self.api_key}"

    def _cached_token(self) -> str | None:
        """Tokens are valid for the trading day only."""
        if not self.token_file or not self.token_file.exists():
            return None
        try:
            blob = json.loads(self.token_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if blob.get("date") != date.today().isoformat():
            return None
        return blob.get("access_token")

    def _cache_token(self, token: str) -> None:
        if not self.token_file:
            return
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(
            json.dumps({"access_token": token, "date": date.today().isoformat()}), encoding="utf-8"
        )

    def exchange_request_token(self, request_token: str) -> str:
        """Exchange a request_token from the login redirect for an access_token."""
        from kiteconnect import KiteConnect  # imported lazily

        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)
        self.access_token = data["access_token"]
        self._cache_token(self.access_token)
        return self.access_token

    @property
    def kite(self) -> Any:
        if self._kite is not None:
            return self._kite
        from kiteconnect import KiteConnect

        token = self.access_token or self._cached_token()
        if not token:
            raise KiteAuthError(
                "No valid Kite access token for today.\n"
                f"  1) Open: {self.login_url()}\n"
                "  2) After login, copy request_token from the redirect URL\n"
                "  3) Run: python scripts/fetch_data.py --request-token <token>"
            )
        kite = KiteConnect(api_key=self.api_key)
        kite.set_access_token(token)
        self.access_token = token
        self._kite = kite
        return kite


# --------------------------------------------------------------------------- #
# Price cache
# --------------------------------------------------------------------------- #
class PriceStore:
    """Parquet-backed OHLCV cache, one file per instrument token."""

    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir) / "prices"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, token: int) -> Path:
        return self.root / f"{token}.parquet"

    def has(self, token: int) -> bool:
        return self._path(token).exists()

    def read(self, token: int) -> pd.DataFrame:
        p = self._path(token)
        if not p.exists():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.read_parquet(p)
        return df.sort_values("date").reset_index(drop=True)

    def write(self, token: int, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df.to_parquet(self._path(token), index=False)

    def upsert(self, token: int, df: pd.DataFrame) -> pd.DataFrame:
        merged = pd.concat([self.read(token), df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        merged = merged.reset_index(drop=True)
        self.write(token, merged)
        return merged

    def last_date(self, token: int) -> pd.Timestamp | None:
        df = self.read(token)
        return None if df.empty else pd.Timestamp(df["date"].iloc[-1])

    def tokens(self) -> list[int]:
        return sorted(int(p.stem) for p in self.root.glob("*.parquet"))


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class KiteDataClient:
    """Read-only historical + live data access with caching."""

    def __init__(self, cfg: AppConfig, session: KiteSession | None = None):
        self.cfg = cfg
        self.store = PriceStore(cfg.paths.data_dir)
        self._session = session
        self._rate = float(cfg.get("data_sources.prices.rate_limit_per_sec", 3))
        self._last_call = 0.0

    @property
    def session(self) -> KiteSession:
        if self._session is None:
            self._session = KiteSession.from_env(self.cfg.paths.data_dir)
        return self._session

    def _throttle(self) -> None:
        if self._rate <= 0:
            return
        gap = 1.0 / self._rate
        elapsed = time.monotonic() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.monotonic()

    # -- instruments -----------------------------------------------------------
    def fetch_instruments(self, exchange: str = "NSE", refresh: bool = False) -> pd.DataFrame:
        cache = self.cfg.paths.data_dir / f"instruments_{exchange}.parquet"
        if cache.exists() and not refresh:
            return pd.read_parquet(cache)
        self._throttle()
        df = pd.DataFrame(self.session.kite.instruments(exchange))
        df.to_parquet(cache, index=False)
        log.info("Fetched %d %s instruments", len(df), exchange)
        return df

    # -- historical ------------------------------------------------------------
    def fetch_historical(
        self,
        token: int,
        start: date | str,
        end: date | str | None = None,
        interval: str = "day",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch candles, paginating around Kite's 2000-day cap.

        With use_cache=True only the missing tail is requested.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp(date.today())

        if use_cache:
            last = self.store.last_date(token)
            if last is not None and last >= end_ts:
                cached = self.store.read(token)
                return cached[(cached["date"] >= start_ts) & (cached["date"] <= end_ts)]
            if last is not None:
                start_ts = max(start_ts, last + pd.Timedelta(days=1))

        if start_ts > end_ts:
            return self.store.read(token)

        frames: list[pd.DataFrame] = []
        for lo, hi in _date_windows(start_ts, end_ts, MAX_DAYS_PER_REQUEST):
            chunk = self._fetch_window(token, lo, hi, interval)
            if chunk is not None and not chunk.empty:
                frames.append(chunk)

        if not frames:
            return self.store.read(token)

        fresh = pd.concat(frames, ignore_index=True)
        return self.store.upsert(token, fresh) if use_cache else fresh

    def _fetch_window(
        self, token: int, lo: pd.Timestamp, hi: pd.Timestamp, interval: str, retries: int = 3
    ) -> pd.DataFrame | None:
        for attempt in range(retries):
            try:
                self._throttle()
                raw = self.session.kite.historical_data(
                    instrument_token=token,
                    from_date=lo.to_pydatetime(),
                    to_date=hi.to_pydatetime(),
                    interval=interval,
                )
                if not raw:
                    return None
                df = pd.DataFrame(raw)
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                return df[[c for c in OHLCV_COLUMNS if c in df.columns]]
            except Exception as exc:  # noqa: BLE001 - network/API errors vary
                wait = 2 ** attempt
                log.warning(
                    "historical_data failed (token=%s %s..%s, attempt %d/%d): %s; retrying in %ds",
                    token, lo.date(), hi.date(), attempt + 1, retries, exc, wait,
                )
                time.sleep(wait)
        log.error("Giving up on token=%s window %s..%s", token, lo.date(), hi.date())
        return None

    def fetch_many(
        self,
        tokens: Iterable[int],
        start: date | str,
        end: date | str | None = None,
        interval: str = "day",
    ) -> dict[int, pd.DataFrame]:
        out: dict[int, pd.DataFrame] = {}
        tokens = list(tokens)
        for i, t in enumerate(tokens, 1):
            try:
                out[t] = self.fetch_historical(t, start, end, interval)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed token %s: %s", t, exc)
            if i % 25 == 0:
                log.info("Fetched %d/%d instruments", i, len(tokens))
        return out

    # -- live / account (read-only) -------------------------------------------
    def holdings(self) -> pd.DataFrame:
        return pd.DataFrame(self.session.kite.holdings())

    def positions(self) -> pd.DataFrame:
        pos = self.session.kite.positions()
        return pd.DataFrame(pos.get("net", []))

    def quote(self, instruments: list[str]) -> dict[str, Any]:
        return self.session.kite.quote(instruments)

    def orders(self) -> pd.DataFrame:
        return pd.DataFrame(self.session.kite.orders())

    def trades(self) -> pd.DataFrame:
        """Executed trades - the raw input for the execution journal."""
        return pd.DataFrame(self.session.kite.trades())


def _date_windows(
    start: pd.Timestamp, end: pd.Timestamp, max_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end] into <= max_days chunks."""
    windows, cur = [], start
    while cur <= end:
        hi = min(cur + timedelta(days=max_days - 1), end)
        windows.append((cur, hi))
        cur = hi + timedelta(days=1)
    return windows


# --------------------------------------------------------------------------- #
# Panel assembly
# --------------------------------------------------------------------------- #
def build_price_panel(
    store: PriceStore,
    token_to_isin: dict[int, str],
    field: str = "close",
) -> pd.DataFrame:
    """Wide panel: index=date, columns=isin, values=`field`."""
    series = {}
    for token, isin in token_to_isin.items():
        df = store.read(token)
        if df.empty or field not in df.columns:
            continue
        s = df.set_index("date")[field]
        s.index = pd.to_datetime(s.index)
        series[isin] = s[~s.index.duplicated(keep="last")]
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index()
    panel.index.name = "date"
    return panel
