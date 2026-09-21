"""Trading-rule signals: rank buffer buy/sell/hold/watch."""

from __future__ import annotations

import pandas as pd

from src.presets import get_preset
from src.signals.trading_rules import SIGNAL_COLUMNS, build_trade_signals


def _rankings(rows: list[tuple]) -> pd.DataFrame:
    data = []
    for isin, rank, composite, mom, quality, value in rows:
        data.append({
            "isin": isin, "symbol": isin, "sector": "Test",
            "rank": rank, "composite": composite,
            "momentum": mom, "quality": quality, "value": value, "low_vol": 0.0,
        })
    return pd.DataFrame(data)


def _cfg(top_n=5, buffer=2.0, mom_w=0.3, q_gate=True):
    cfg = get_preset("price_only_core")
    cfg.sleeve_a.top_n = top_n
    cfg.sleeve_a.rank_buffer_multiple = buffer
    cfg.sleeve_a.factors.weight_momentum = mom_w
    cfg.sleeve_a.factors.value_quality_gate = q_gate
    return cfg


def _core(n=8):
    """AAA.. names ranked 1..n with positive momentum."""
    rows = []
    for i in range(n):
        name = chr(ord("A") * 0 + 65) + str(i)  # A0, A1, ...
        name = f"N{i:02d}"
        rows.append((name, i + 1, 2.0 - i * 0.15, 1.0 - i * 0.05, None, None))
    return rows


def test_new_name_in_top_n_with_room_is_a_buy():
    # Hold the first 4 of top 5; N04 (rank 5) should buy.
    ranks = _rankings(_core(8))
    held = {f"N{i:02d}" for i in range(4)}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    assert set(out.by_action("BUY")["isin"]) == {"N04"}
    assert out.summary["n_buys"] == 1
    assert out.summary["n_room"] == 1


def test_held_name_past_buffer_is_a_sell():
    # top_n=5, buffer=2 → sell after rank 10
    ranks = _rankings(_core(8) + [("OLD", 12, -1.0, -0.5, None, None)])
    held = {f"N{i:02d}" for i in range(4)} | {"OLD"}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    assert set(out.by_action("SELL")["isin"]) == {"OLD"}
    assert "N04" in set(out.by_action("BUY")["isin"])


def test_held_name_inside_buffer_is_held():
    ranks = _rankings(_core(8))
    # N07 is rank 8: outside top 5, inside buffer 10
    held = {"N00", "N01", "N07"}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    assert "N07" in set(out.by_action("HOLD")["isin"])
    assert out.summary["n_sells"] == 0


def test_book_full_makes_entry_zone_a_watch():
    ranks = _rankings(_core(8))
    held = {f"N{i:02d}" for i in range(5)}  # all 5 slots filled, all inside buffer
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    assert out.summary["n_buys"] == 0
    assert out.summary["n_room"] == 0
    # N05 is rank 6, buffer zone watch — not entry. Swap so a new name is rank 5.
    ranks.loc[ranks["isin"] == "N05", "rank"] = 5
    ranks.loc[ranks["isin"] == "N04", "rank"] = 6
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    watch = out.by_action("WATCH")
    assert "N05" in set(watch["isin"])
    assert "book_full" in str(watch.loc[watch["isin"] == "N05", "rules"].iloc[0])
    assert out.summary["n_buys"] == 0


def test_negative_momentum_blocks_a_new_buy():
    ranks = _rankings(_core(8))
    ranks.loc[ranks["isin"] == "N04", "momentum"] = -0.4
    held = {f"N{i:02d}" for i in range(4)}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    assert "N04" in set(out.by_action("WATCH")["isin"])
    assert out.summary["n_buys"] == 0


def test_value_quality_gate_blocks_cheap_junk():
    cfg = _cfg(mom_w=0.0, q_gate=True)
    ranks = _rankings(_core(8))
    ranks["momentum"] = None
    ranks["quality"] = 0.8
    ranks["value"] = 0.2
    ranks.loc[ranks["isin"] == "N04", "quality"] = -0.8
    ranks.loc[ranks["isin"] == "N04", "value"] = 1.5
    held = {f"N{i:02d}" for i in range(4)}
    out = build_trade_signals(ranks, held_isins=held, config=cfg)
    junk = out.frame[out.frame["isin"] == "N04"].iloc[0]
    assert junk["action"] == "WATCH"
    assert "value_quality_gate" in junk["rules"]


def test_etf_uses_dual_momentum_and_own_book():
    ranks = _rankings(_core(8))
    ranks.loc[ranks["isin"] == "N04", ["isin", "symbol", "asset_class", "abs_momentum"]] = (
        "INF204KB14I2", "NIFTYBEES", "etf", 0.12
    )
    held = {f"N{i:02d}" for i in range(4)}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    bees = out.frame[out.frame["symbol"] == "NIFTYBEES"]
    assert not bees.empty
    assert bees.iloc[0]["asset_class"] == "etf"


def test_mf_folio_is_satellite_hold():
    ranks = _rankings(_core(8))
    ranks.loc[ranks["isin"] == "N07", ["isin", "symbol", "asset_class"]] = (
        "INF090I01239", "HDFC_EQ", "mf"
    )
    held = {"INF090I01239"}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    row = out.frame[out.frame["symbol"] == "HDFC_EQ"].iloc[0]
    assert row["action"] == "HOLD"
    assert "mf_satellite" in row["rules"]


def test_empty_rankings_is_safe():
    out = build_trade_signals(pd.DataFrame(), held_isins=set(), config=_cfg())
    assert out.summary["n_buys"] == 0
    assert list(out.frame.columns) == list(SIGNAL_COLUMNS)


def test_negative_profitability_blocks_a_new_buy():
    ranks = _rankings(_core(8))
    ranks["profitability"] = 0.8
    ranks.loc[ranks["isin"] == "N04", "profitability"] = -0.3
    held = {f"N{i:02d}" for i in range(4)}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg(mom_w=0.0, q_gate=False))
    row = out.frame[out.frame["isin"] == "N04"].iloc[0]
    assert row["action"] == "WATCH"
    assert "profitability_failed" in row["rules"]
    assert out.summary["n_buys"] == 0


def test_profitability_break_sells_inside_buffer():
    """Leave top-N with a negative IIMA profitability print → close, don't wait for buffer."""
    ranks = _rankings(_core(8))
    ranks["profitability"] = 0.5
    ranks.loc[ranks["isin"] == "N07", "profitability"] = -0.4
    held = {"N00", "N01", "N07"}
    out = build_trade_signals(ranks, held_isins=held, config=_cfg())
    row = out.frame[out.frame["isin"] == "N07"].iloc[0]
    assert row["action"] == "SELL"
    assert "profitability_broke" in row["rules"]
    assert "N01" in set(out.by_action("HOLD")["isin"])
