"""Databento-style real-time price-movement scanner (Indian book).

The Databento tutorial keeps three O(1) dictionaries and never walks a
DataFrame on the tick path:

  * ``symbol_directory``  — instrument token → ticker
  * ``last_day_lookup``   — ticker → previous close
  * ``is_signal_lit``     — ticker → already alerted this session

A name lights once when ``abs(last / prev_close - 1)`` first exceeds the
threshold (default 3%, same as the article). This module does not place
orders; it only emits timestamped alerts for the dashboard tape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

DEFAULT_THRESHOLD = 0.03


@dataclass(frozen=True)
class MovementAlert:
    symbol: str
    ts: datetime
    last: float
    previous: float
    abs_return: float

    def format(self) -> str:
        ts = self.ts.astimezone().isoformat(timespec="milliseconds")
        return (
            f"[{ts}] {self.symbol} moved by {self.abs_return * 100:.2f}% "
            f"(current: {self.last:.4f}, previous: {self.previous:.4f})"
        )


@dataclass
class PriceMovementScanner:
    """Scan a live feed for large moves versus yesterday's close."""

    pct_threshold: float = DEFAULT_THRESHOLD
    symbol_directory: dict[int, str] = field(default_factory=dict)
    last_day_lookup: dict[str, float] = field(default_factory=dict)
    is_signal_lit: dict[str, bool] = field(default_factory=dict)

    def register(self, symbol: str, prev_close: float, token: int | None = None) -> None:
        sym = symbol.strip().upper()
        if not sym or prev_close <= 0:
            return
        self.last_day_lookup[sym] = float(prev_close)
        self.is_signal_lit.setdefault(sym, False)
        if token is not None:
            self.symbol_directory[int(token)] = sym

    def register_many(self, rows: Iterable[tuple[str, float, int | None]]) -> None:
        for symbol, prev_close, token in rows:
            self.register(symbol, prev_close, token)

    def reset_latches(self) -> None:
        for key in self.is_signal_lit:
            self.is_signal_lit[key] = False

    def scan(
        self,
        symbol: str | None = None,
        last: float | None = None,
        token: int | None = None,
        ts: datetime | None = None,
    ) -> MovementAlert | None:
        """Hot path: dict lookups only. Returns an alert the first time a name crosses."""
        if token is not None and not symbol:
            symbol = self.symbol_directory.get(int(token))
        if not symbol or last is None or last <= 0:
            return None
        sym = symbol.strip().upper()
        prev = self.last_day_lookup.get(sym)
        if prev is None or prev <= 0:
            return None
        abs_r = abs(last - prev) / prev
        if abs_r <= self.pct_threshold or self.is_signal_lit.get(sym):
            return None
        self.is_signal_lit[sym] = True
        return MovementAlert(
            symbol=sym,
            ts=ts or datetime.now(timezone.utc),
            last=float(last),
            previous=float(prev),
            abs_return=abs_r,
        )
