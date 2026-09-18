"""Rank buffering (hysteresis) for turnover control.

Naive top-N rebalancing churns badly: a stock oscillating around rank N is
bought and sold repeatedly, paying costs and short-term capital gains tax each
time for no signal benefit.

The buffer rule: **buy** when a stock enters the top N, but only **sell** once
it falls past `buffer_multiple * N`. This is standard in index and factor
construction and typically cuts turnover substantially with minimal alpha loss
- a meaningful after-tax win under India's STCG regime.
"""

from __future__ import annotations

import pandas as pd


def apply_rank_buffer(
    scores: pd.Series,
    current_holdings: set[str] | None,
    top_n: int,
    buffer_multiple: float = 2.0,
) -> list[str]:
    """Select names using buy/sell hysteresis.

    Returns the target holdings (at most `top_n`), preferring incumbents that
    remain inside the buffer zone before adding new entrants.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    if ranked.empty:
        return []

    top_n = max(1, min(top_n, len(ranked)))
    buffer_n = min(len(ranked), int(round(top_n * max(1.0, buffer_multiple))))

    entry_set = set(ranked.index[:top_n])
    buffer_set = set(ranked.index[:buffer_n])
    current = set(current_holdings or set())

    # Incumbents survive while inside the wider buffer band.
    keep = [n for n in ranked.index[:buffer_n] if n in current and n in buffer_set]

    selected = list(keep)
    if len(selected) < top_n:
        for name in ranked.index:
            if len(selected) >= top_n:
                break
            if name not in selected and name in entry_set:
                selected.append(name)
    # If incumbents alone exceed top_n, keep the best-ranked ones.
    if len(selected) > top_n:
        selected = [n for n in ranked.index if n in set(selected)][:top_n]

    return selected


def turnover(previous: pd.Series, current: pd.Series) -> float:
    """One-way turnover between two weight vectors (0.5 * sum |dw|)."""
    if previous is None or previous.empty:
        return float(current.sum()) if not current.empty else 0.0
    idx = previous.index.union(current.index)
    p = previous.reindex(idx).fillna(0.0)
    c = current.reindex(idx).fillna(0.0)
    return float((c - p).abs().sum() / 2.0)


def holdings_overlap(previous: set[str], current: set[str]) -> float:
    """Fraction of the new portfolio already held."""
    if not current:
        return 1.0
    return len(previous & current) / len(current)
