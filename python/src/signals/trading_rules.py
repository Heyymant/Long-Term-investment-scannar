"""Turn factor rankings into an actionable BUY / SELL / HOLD / WATCH list.

The rules match Sleeve A construction (`apply_rank_buffer`):

  * **BUY**  — not held, rank ≤ top_N, and a slot is free after sells.
  * **SELL** — held, and rank has fallen past ``top_N × buffer_multiple``.
  * **HOLD** — held, and still inside the buffer band.
  * **WATCH** — in the entry/buffer zone but not selected (book full, or a
    confirmation rule failed).

Extra confirmation rules (skipped when the data is missing, e.g. a
price-only run with no quality/value scores):

  * Momentum confirmation: a new buy needs positive momentum.
  * Value × Quality gate: cheap-and-junk names are not bought.

This module never places orders. It writes a checklist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..schema import StrategyConfig

SIGNAL_COLUMNS = [
    "symbol", "isin", "action", "sleeve", "rank", "composite",
    "quality", "value", "momentum", "low_vol",
    "sector", "held", "weight", "conviction", "rules", "reason",
    "asset_class",
]


@dataclass
class SignalsReport:
    frame: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)

    def by_action(self, action: str) -> pd.DataFrame:
        if self.frame.empty:
            return self.frame
        return self.frame[self.frame["action"] == action].copy()


def build_trade_signals(
    rankings: pd.DataFrame,
    held_isins: set[str] | None,
    config: StrategyConfig,
    weights: pd.Series | dict[str, float] | None = None,
    event_signals: pd.DataFrame | None = None,
    statarb_signals: pd.DataFrame | None = None,
) -> SignalsReport:
    """Score every ranked name and keep the actionable rows."""
    if rankings is None or rankings.empty:
        return SignalsReport(
            pd.DataFrame(columns=SIGNAL_COLUMNS),
            {"n_buys": 0, "n_sells": 0, "note": "no rankings"},
        )

    df = rankings.copy()
    if "isin" not in df.columns and df.index.name == "isin":
        df = df.reset_index()
    if "rank" not in df.columns and "composite" in df.columns:
        df["rank"] = df["composite"].rank(ascending=False)

    sleeve = config.sleeve_a
    top_n = int(sleeve.top_n)
    buffer_n = int(round(top_n * max(1.0, float(sleeve.rank_buffer_multiple))))
    held = {str(x) for x in (held_isins or set())}
    weight_map = _as_weight_map(weights)

    n_scored = int(df["isin"].nunique()) if "isin" in df.columns else len(df)
    buffer_n = min(buffer_n, max(top_n, n_scored))

    mom_on = float(sleeve.factors.weight_momentum) > 0
    lv_on = float(sleeve.factors.weight_low_vol) > 0
    q_gate = bool(sleeve.factors.value_quality_gate)
    q_w = float(sleeve.factors.weight_quality)
    has_mom = "momentum" in df.columns and df["momentum"].notna().any()
    has_q = "quality" in df.columns and df["quality"].notna().any()
    has_v = "value" in df.columns and df["value"].notna().any()
    has_lv = "low_vol" in df.columns and df["low_vol"].notna().any()

    from ..data.asset_class import classify
    from ..factors.satellite import ETF_TOP_N

    if "asset_class" not in df.columns:
        df["asset_class"] = [
            classify(str(r.get("isin", "")), str(r.get("symbol", "")))
            for _, r in df.iterrows()
        ]

    etf_top = ETF_TOP_N
    etf_buffer = int(round(etf_top * max(1.0, float(sleeve.rank_buffer_multiple))))
    equity_keep = {
        str(r["isin"]) for _, r in df.iterrows()
        if str(r.get("asset_class") or "equity") == "equity"
        and str(r["isin"]) in held and float(r["rank"]) <= buffer_n
    }
    etf_keep = {
        str(r["isin"]) for _, r in df.iterrows()
        if str(r.get("asset_class")) == "etf"
        and str(r["isin"]) in held and float(r["rank"]) <= etf_buffer
    }
    keep = equity_keep
    room = max(0, top_n - len(equity_keep))
    etf_room = max(0, etf_top - len(etf_keep))

    buy_slots: list[str] = []
    etf_slots: list[str] = []
    rows: list[dict] = []

    ranked = df.sort_values("rank")
    for _, r in ranked.iterrows():
        isin = str(r["isin"])
        rank = float(r["rank"])
        is_held = isin in held
        symbol = str(r["symbol"] if pd.notna(r.get("symbol")) else isin)
        sector = str(r["sector"] if pd.notna(r.get("sector")) else "UNKNOWN")
        asset = str(r.get("asset_class") or classify(isin, symbol))
        composite = _num(r.get("composite"))
        quality = _num(r.get("quality"))
        value = _num(r.get("value"))
        momentum = _num(r.get("momentum"))
        low_vol = _num(r.get("low_vol"))
        abs_mom = _num(r.get("abs_momentum"))

        sleeve_tag = "ETF" if asset == "etf" else "A"
        use_top = etf_top if asset == "etf" else top_n
        use_buf = etf_buffer if asset == "etf" else buffer_n
        use_room = etf_room if asset == "etf" else room
        slots = etf_slots if asset == "etf" else buy_slots

        rules: list[str] = []
        action = None
        reason = ""

        if asset == "mf":
            action = "HOLD" if is_held else "WATCH"
            rules.append("mf_satellite")
            reason = (
                "Mutual-fund folio: satellite holding. Prefer a listed ETF "
                "when momentum and low-vol agree."
            )
        elif is_held:
            profitability = _num(r.get("profitability"))
            thesis_dead = (
                asset == "equity"
                and profitability is not None
                and profitability < 0
                and rank > use_top
            )
            if rank > use_buf or thesis_dead:
                action = "SELL"
                if rank > use_buf:
                    rules.append("rank_buffer_exit")
                    reason = (
                        f"Rank {rank:.0f} is past the sell line ({use_buf}). "
                        "The buffer no longer protects this holding."
                    )
                else:
                    rules.append("profitability_broke")
                    reason = (
                        f"Rank {rank:.0f} left the top {use_top} and IIMA "
                        "profitability turned negative. Close — the quality thesis is broken."
                    )
            else:
                action = "HOLD"
                rules.append("rank_buffer_keep")
                if rank <= use_top:
                    rules.append("still_in_entry_zone")
                    reason = f"Rank {rank:.0f} is still inside the top {use_top}. Keep."
                else:
                    reason = (
                        f"Rank {rank:.0f} is outside the top {use_top} but inside "
                        f"the buffer (sell only after {use_buf}). Keep."
                    )
        elif (not is_held) and rank <= use_top:
            blocked, block_rules, block_reason = _confirmation_blocks(
                momentum, quality, value, low_vol, abs_mom,
                profitability=_num(r.get("profitability")),
                mom_on=mom_on and has_mom,
                q_gate=q_gate and has_q and has_v and asset == "equity",
                quality_on=q_w > 0 and has_q and asset == "equity",
                low_vol_on=lv_on and has_lv,
                dual_on=asset == "etf",
            )
            if blocked:
                action = "WATCH"
                rules.extend(block_rules)
                reason = block_reason
            elif len(slots) < use_room:
                action = "BUY"
                rules.append("rank_buffer_entry")
                if asset == "etf":
                    rules.append("dual_momentum")
                slots.append(isin)
                left = use_room - len(slots)
                reason = (
                    f"Rank {rank:.0f} entered the top {use_top} and a slot is free"
                    + (f" ({left} more after this name)." if left else ".")
                )
            else:
                action = "WATCH"
                rules.append("book_full")
                reason = (
                    f"Rank {rank:.0f} is in the entry zone but the book is full. "
                    "Wait for a sell."
                )
        elif (not is_held) and rank <= use_buf:
            action = "WATCH"
            rules.append("buffer_zone")
            reason = (
                f"Rank {rank:.0f} is inside the buffer band (≤ {use_buf}) "
                f"but not yet in the top {use_top}."
            )
        else:
            continue

        conviction = _conviction(action, composite, rank, use_top, use_buf)
        rows.append({
            "symbol": symbol,
            "isin": isin,
            "action": action,
            "sleeve": sleeve_tag,
            "rank": int(rank),
            "composite": composite,
            "quality": quality,
            "value": value,
            "momentum": momentum,
            "low_vol": low_vol,
            "sector": sector,
            "held": is_held,
            "weight": weight_map.get(isin),
            "conviction": conviction,
            "rules": ";".join(rules),
            "reason": reason,
            "asset_class": asset,
        })

    extra = _sleeve_overlays(event_signals, statarb_signals, held, {r["isin"] for r in rows})
    rows.extend(extra)

    frame = pd.DataFrame(rows, columns=SIGNAL_COLUMNS)
    if not frame.empty:
        order = {"SELL": 0, "BUY": 1, "WATCH": 2, "HOLD": 3}
        frame["_ord"] = frame["action"].map(order)
        frame = frame.sort_values(
            ["_ord", "conviction"], ascending=[True, False]
        ).drop(columns="_ord").reset_index(drop=True)

    summary = {
        "as_of_ranks": n_scored,
        "top_n": top_n,
        "buffer_n": buffer_n,
        "n_held": len(held),
        "n_keep": len(keep),
        "n_room": room,
        "n_buys": int((frame["action"] == "BUY").sum()) if not frame.empty else 0,
        "n_sells": int((frame["action"] == "SELL").sum()) if not frame.empty else 0,
        "n_holds": int((frame["action"] == "HOLD").sum()) if not frame.empty else 0,
        "n_watch": int((frame["action"] == "WATCH").sum()) if not frame.empty else 0,
        "n_queued": int(
            ((frame["action"] == "WATCH") & (frame["rank"].fillna(999) <= top_n)).sum()
        ) if not frame.empty else 0,
        "etf_top_n": etf_top,
        "etf_buffer_n": etf_buffer,
        "n_etf_keep": len(etf_keep),
        "n_etf_room": etf_room,
        "rules": _rule_copy(
            top_n, buffer_n, mom_on and has_mom, q_gate and has_q and has_v,
            quality_on=q_w > 0 and has_q, low_vol_on=lv_on and has_lv,
        ),
        "note": "Decision support only — no orders are placed. Execute manually.",
    }
    return SignalsReport(frame, summary)


def signals_from_artifacts(
    run_dir: str | Path,
    config: StrategyConfig | None = None,
) -> SignalsReport:
    """Rebuild the signal list from a run's JSON artifacts (no market reload)."""
    run_dir = Path(run_dir)
    cfg = config or StrategyConfig.load(run_dir / "strategy_config.json")
    rankings = _read_table(run_dir / "rankings.json")
    holdings = _read_table(run_dir / "holdings_target.json")
    held = set()
    weights = None
    if not holdings.empty and "isin" in holdings.columns:
        held = set(holdings["isin"].astype(str))
        if "weight" in holdings.columns:
            weights = pd.Series(holdings["weight"].values, index=holdings["isin"].astype(str))
    events = _read_table(run_dir / "event_signals.json")
    statarb = _read_table(run_dir / "statarb_signals.json")
    return build_trade_signals(
        rankings, held, cfg, weights,
        event_signals=events if not events.empty else None,
        statarb_signals=statarb if not statarb.empty else None,
    )


def _confirmation_blocks(
    momentum: float | None,
    quality: float | None,
    value: float | None,
    low_vol: float | None = None,
    abs_mom: float | None = None,
    profitability: float | None = None,
    *,
    mom_on: bool,
    q_gate: bool,
    quality_on: bool = False,
    low_vol_on: bool = False,
    dual_on: bool = False,
) -> tuple[bool, list[str], str]:
    if profitability is not None and profitability < 0:
        return True, ["profitability_failed"], (
            "IIMA profitability is negative. Do not open a new long — "
            "in India that print is the tunnelling signal."
        )
    if mom_on and momentum is not None and momentum <= 0:
        return True, ["momentum_failed"], (
            "In the entry zone, but 12–1 / 6–1 momentum is non-positive. "
            "Wait for trend confirmation."
        )
    if dual_on and abs_mom is not None and abs_mom <= 0:
        return True, ["dual_momentum_failed"], (
            "Relative momentum is fine, but the 12-month absolute return is "
            "negative (dual momentum / time-series overlay). Stay in cash on this ETF."
        )
    if low_vol_on and low_vol is not None and low_vol < -1.0:
        return True, ["high_vol_block"], (
            "Low-vol score is deeply negative — this name is in the high-vol "
            "bucket that crashes with Indian momentum. Skip until it calms."
        )
    if quality_on and quality is not None and quality < 0:
        return True, ["quality_failed"], (
            "Quality (QMJ) is negative. Indian evidence puts quality first; "
            "do not start a new long here."
        )
    if q_gate and quality is not None and value is not None and value > 0 and quality < 0:
        return True, ["value_quality_gate"], (
            "Cheap (positive value) but poor quality — the value×quality gate "
            "blocks a new buy."
        )
    return False, [], ""


def _conviction(action: str, composite: float | None, rank: float,
                top_n: int, buffer_n: int) -> float:
    if action == "SELL":
        # How far past the sell line, 0 at the line, 1 well beyond.
        span = max(1.0, float(buffer_n))
        return round(min(1.0, max(0.0, (rank - buffer_n) / span)), 4)
    if composite is None:
        return round(max(0.0, 1.0 - (rank - 1) / max(top_n, 1)), 4)
    # Composite z-scores are typically in [-3, 3]; map to (0, 1].
    return round(max(0.0, min(1.0, 0.5 + float(composite) / 4.0)), 4)


def _sleeve_overlays(
    event_signals: pd.DataFrame | None,
    statarb_signals: pd.DataFrame | None,
    held: set[str],
    already: set[str],
) -> list[dict]:
    extra: list[dict] = []
    if event_signals is not None and not event_signals.empty:
        for _, r in event_signals.iterrows():
            isin = str(r.get("isin", ""))
            if not isin or isin in already:
                continue
            extra.append(_overlay_row(
                isin, "C", "BUY" if isin not in held else "HOLD",
                rules="pead_sue",
                reason=str(r.get("entry_reason") or "Positive SUE PEAD entry."),
                held=isin in held,
                extra=r,
            ))
    if statarb_signals is not None and not statarb_signals.empty:
        action_col = "action" if "action" in statarb_signals.columns else None
        for _, r in statarb_signals.iterrows():
            raw = str(r.get(action_col) or "")
            if not raw.startswith("ENTER"):
                continue
            isin = str(r.get("leg") or r.get("isin") or r.get("y") or "")
            if not isin or isin in already:
                continue
            extra.append(_overlay_row(
                isin, "B", "BUY",
                rules="statarb_entry",
                reason=f"Stat-arb {raw} (z={r.get('z_score', 'n/a')}).",
                held=isin in held,
                extra=r,
            ))
    return extra


def _overlay_row(isin, sleeve, action, rules, reason, held, extra) -> dict:
    return {
        "symbol": str(extra.get("symbol") or isin),
        "isin": isin,
        "action": action,
        "sleeve": sleeve,
        "rank": _num(extra.get("rank")),
        "composite": _num(extra.get("sue") or extra.get("z_score")),
        "quality": None, "value": None, "momentum": None, "low_vol": None,
        "sector": str(extra.get("sector") or "UNKNOWN"),
        "held": held,
        "weight": _num(extra.get("weight")),
        "conviction": 0.6,
        "rules": rules,
        "reason": reason,
        "asset_class": "equity",
    }


def _rule_copy(
    top_n: int, buffer_n: int, mom_on: bool, q_gate: bool,
    quality_on: bool = False, low_vol_on: bool = False,
) -> list[dict]:
    rules = [
        {
            "id": "rank_buffer_entry",
            "action": "BUY",
            "text": f"Buy names that enter the top {top_n} when a slot is free.",
        },
        {
            "id": "rank_buffer_exit",
            "action": "SELL",
            "text": f"Sell a holding only after its rank falls past {buffer_n} "
                    f"(top {top_n} × buffer). Oscillation around {top_n} is ignored.",
        },
        {
            "id": "rank_buffer_keep",
            "action": "HOLD",
            "text": f"Keep holdings ranked {top_n + 1}–{buffer_n}. The buffer exists "
                    "to cut turnover and STCG.",
        },
    ]
    if mom_on:
        rules.append({
            "id": "momentum_failed",
            "action": "WATCH",
            "text": "A new buy needs positive 12–1 / 6–1 momentum (skip-month, India).",
        })
    if quality_on:
        rules.append({
            "id": "quality_failed",
            "action": "WATCH",
            "text": "Quality (QMJ) must be non-negative to open a new equity long.",
        })
    if q_gate:
        rules.append({
            "id": "value_quality_gate",
            "action": "WATCH",
            "text": "Do not buy cheap-and-junk names (positive value, negative quality).",
        })
    rules.append({
        "id": "profitability_broke",
        "action": "SELL",
        "text": "Close a holding that has left the top-N if IIMA profitability turns negative "
                "(quality thesis broken). Do not use a percent stop-loss — that cuts winners.",
    })
    rules.append({
        "id": "trend_overlay",
        "action": "HOLD",
        "text": "Scale the whole book with the 200-day MA (Moskowitz absolute momentum, ±2% hysteresis). "
                "This is a portfolio overlay, not a stock-level stop.",
    })
    rules.append({
        "id": "tax_defer",
        "action": "HOLD",
        "text": "Defer a rank-buffer sell if the lot is inside 30 days of 12-month LTCG "
                "(STCG 20% vs LTCG 12.5%), unless rank is already past the sell line.",
    })
    if low_vol_on:
        rules.append({
            "id": "high_vol_block",
            "action": "WATCH",
            "text": "Skip names with a deeply negative low-vol score (momentum crash filter).",
        })
    rules.append({
        "id": "dual_momentum",
        "action": "BUY",
        "text": "ETFs need dual momentum: relative rank plus a positive 12-month return.",
    })
    rules.append({
        "id": "mf_satellite",
        "action": "HOLD",
        "text": "Mutual-fund folios stay as satellite holdings; prefer a listed ETF to add.",
    })
    rules.append({
        "id": "book_full",
        "action": "WATCH",
        "text": "Entry-zone names wait if every current holding is still inside the buffer.",
    })
    return rules


def _as_weight_map(weights) -> dict[str, float]:
    if weights is None:
        return {}
    if isinstance(weights, pd.Series):
        return {str(k): float(v) for k, v in weights.items() if pd.notna(v)}
    return {str(k): float(v) for k, v in dict(weights).items()}


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    cols = payload.get("columns") or []
    rows = payload.get("rows") or []
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=cols)
