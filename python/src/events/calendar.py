"""Earnings calendar views for the dashboard Events tab."""

from __future__ import annotations

import pandas as pd

from ..data.earnings import EarningsCalendar


def build_calendar_view(
    calendar: EarningsCalendar,
    sue_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 30,
    lookahead_days: int = 45,
    securities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Recent and upcoming results, annotated with the latest SUE."""
    ts = pd.Timestamp(as_of)
    if calendar.data.empty:
        return pd.DataFrame()

    d = pd.to_datetime(calendar.data["announce_date"])
    window = calendar.data[
        (d >= ts - pd.Timedelta(days=lookback_days))
        & (d <= ts + pd.Timedelta(days=lookahead_days))
    ].copy()
    if window.empty:
        return pd.DataFrame()

    window["announce_date"] = pd.to_datetime(window["announce_date"])
    window["status"] = window["announce_date"].apply(
        lambda x: "reported" if x <= ts else "upcoming"
    )
    window["days_from_now"] = (window["announce_date"] - ts).dt.days

    if not sue_panel.empty:
        latest_sue = (
            sue_panel.sort_values("announce_date")
            .groupby("isin")
            .tail(1)
            .set_index("isin")["sue"]
        )
        window["sue"] = window["isin"].map(latest_sue)

    if securities is not None and not securities.empty:
        name_map = securities.set_index("isin")
        for col in ("symbol", "name", "sector"):
            if col in name_map.columns:
                window[col] = window["isin"].map(name_map[col])

    cols = [c for c in ("isin", "symbol", "name", "sector", "announce_date", "status",
                        "days_from_now", "fiscal_quarter", "eps", "sue")
            if c in window.columns]
    return window[cols].sort_values("announce_date").reset_index(drop=True)


def earnings_season_intensity(
    calendar: EarningsCalendar, freq: str = "W"
) -> pd.Series:
    """How many companies report per period - identifies season peaks."""
    if calendar.data.empty:
        return pd.Series(dtype=int)
    d = pd.to_datetime(calendar.data["announce_date"])
    return d.dt.to_period(freq).value_counts().sort_index()
