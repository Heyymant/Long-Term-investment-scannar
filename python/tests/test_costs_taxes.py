"""Cost and tax model tests with golden numbers.

These are the figures that decide whether a strategy is viable after
frictions, so they are pinned explicitly rather than asserted loosely.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.costs import CostModel, breakeven_cost_bps, capacity_estimate
from src.backtest.taxes import TaxConfig, TaxLotBook, tax_aware_sell_priority


@pytest.fixture
def costs():
    return CostModel(
        brokerage_pct=0.0, stt_buy_pct=0.001, stt_sell_pct=0.001,
        exchange_txn_pct=0.0000297, sebi_charges_pct=0.000001,
        stamp_duty_buy_pct=0.00015, gst_pct=0.18,
        dp_charge_per_scrip_sell=15.34, slippage_bps=5.0, impact_model="none",
    )


# --------------------------------------------------------------------- costs
def test_buy_charges_include_stt_and_stamp(costs):
    c = costs.statutory_charges(100_000, is_buy=True)
    assert c["stt"] == pytest.approx(100.0)        # 0.1%
    assert c["stamp"] == pytest.approx(15.0)       # 0.015%, buy side only
    assert c["gst"] > 0
    assert c["total"] == pytest.approx(100 + 2.97 + 0.1 + 15 + (2.97 + 0.1) * 0.18, rel=1e-6)


def test_sell_has_no_stamp_duty_but_adds_dp(costs):
    charges = costs.statutory_charges(100_000, is_buy=False)
    assert charges["stamp"] == 0.0

    trade = costs.trade_cost(100_000, is_buy=False, new_scrip_sale=True)
    assert trade["dp"] == pytest.approx(15.34)


def test_slippage_scales_with_participation():
    """Square-root impact: larger orders must cost proportionally more."""
    model = CostModel(slippage_bps=5.0, impact_model="sqrt", impact_coefficient=0.1)
    adv = 10_000_000.0
    small = model.slippage_cost(100_000, adv) / 100_000
    large = model.slippage_cost(5_000_000, adv) / 5_000_000
    assert large > small


def test_impact_ignored_without_adv():
    model = CostModel(slippage_bps=5.0, impact_model="sqrt", impact_coefficient=0.1)
    assert model.slippage_cost(100_000, None) == pytest.approx(50.0)


def test_rebalance_cost_aggregates_buys_and_sells(costs):
    trades = pd.Series({"A": 100_000.0, "B": -50_000.0})
    out = costs.rebalance_cost(trades)
    assert out["buy_value"] == 100_000
    assert out["sell_value"] == 50_000
    assert out["total"] > 0
    assert out["bps"] > 0


def test_breakeven_cost_reflects_turnover():
    # 3% excess return with 1x annual turnover -> 150bps round trip.
    assert breakeven_cost_bps(0.13, 0.10, 1.0) == pytest.approx(150.0)
    # Higher turnover erodes the edge faster.
    assert breakeven_cost_bps(0.13, 0.10, 4.0) == pytest.approx(37.5)


def test_capacity_is_bound_by_least_liquid_name():
    weights = pd.Series({"BIG": 0.5, "SMALL": 0.5})
    adv = pd.Series({"BIG": 100_000_000.0, "SMALL": 1_000_000.0})
    cap = capacity_estimate(weights, adv, max_participation=0.1, rebalance_days=5)
    assert cap["binding_isin"] == "SMALL"
    assert cap["capacity_inr"] == pytest.approx(1_000_000 * 0.1 * 5 / 0.5)


# ---------------------------------------------------------------------- tax
@pytest.fixture
def book():
    return TaxLotBook(TaxConfig(
        enabled=True, stcg_rate=0.20, ltcg_rate=0.125,
        ltcg_exemption_per_year=125_000, ltcg_holding_days=365,
    ))


def test_fifo_matches_oldest_lot_first(book):
    book.buy("X", 100, 100.0, pd.Timestamp("2022-01-01"))
    book.buy("X", 100, 200.0, pd.Timestamp("2022-06-01"))
    gains = book.sell("X", 100, 300.0, pd.Timestamp("2022-09-01"))

    assert len(gains) == 1
    assert gains[0].cost_basis == pytest.approx(10_000)   # the 100-rupee lot
    assert gains[0].gain == pytest.approx(20_000)
    assert book.quantity("X") == 100


def test_sell_spanning_lots_splits_correctly(book):
    book.buy("X", 100, 100.0, pd.Timestamp("2022-01-01"))
    book.buy("X", 100, 200.0, pd.Timestamp("2022-06-01"))
    gains = book.sell("X", 150, 300.0, pd.Timestamp("2022-09-01"))

    assert len(gains) == 2
    assert gains[0].quantity == 100
    assert gains[1].quantity == 50
    assert book.quantity("X") == 50


def test_holding_period_determines_stcg_vs_ltcg(book):
    book.buy("S", 100, 100.0, pd.Timestamp("2022-01-01"))
    short = book.sell("S", 100, 150.0, pd.Timestamp("2022-06-01"))
    assert not short[0].is_long_term

    book.buy("L", 100, 100.0, pd.Timestamp("2021-01-01"))
    long = book.sell("L", 100, 150.0, pd.Timestamp("2022-06-01"))
    assert long[0].is_long_term


def test_ltcg_exemption_is_applied(book):
    book.buy("X", 1000, 100.0, pd.Timestamp("2021-01-01"))
    book.sell("X", 1000, 200.0, pd.Timestamp("2022-06-01"))   # 100k gain, long term

    tax = book.tax_for_year(2022)
    # Gain is below the 125k exemption, so no LTCG tax.
    assert tax["ltcg_gain"] == pytest.approx(100_000)
    assert tax["ltcg_tax"] == 0.0


def test_stcg_taxed_at_full_rate(book):
    book.buy("X", 1000, 100.0, pd.Timestamp("2022-05-01"))
    book.sell("X", 1000, 200.0, pd.Timestamp("2022-09-01"))    # 100k short-term

    tax = book.tax_for_year(2022)
    assert tax["stcg_tax"] == pytest.approx(20_000)            # 20%


def test_financial_year_runs_april_to_march(book):
    book.buy("X", 100, 100.0, pd.Timestamp("2022-01-01"))
    book.sell("X", 100, 150.0, pd.Timestamp("2022-02-01"))     # Feb 2022 -> FY2021
    assert book.tax_for_year(2021)["stcg_gain"] == pytest.approx(5_000)
    assert book.tax_for_year(2022)["stcg_gain"] == 0


def test_tax_aware_rebalance_defers_near_ltcg_sales(book):
    # Bought 340 days ago and in profit: selling now costs 20% instead of 12.5%.
    today = pd.Timestamp("2023-01-01")
    book.buy("X", 100, 100.0, today - pd.Timedelta(days=340))

    adjusted = tax_aware_sell_priority(
        book, {"X": 15_000.0}, {"X": 150.0}, today, defer_window_days=45
    )
    assert adjusted["X"] < 15_000.0


def test_tax_aware_does_not_defer_losses(book):
    today = pd.Timestamp("2023-01-01")
    book.buy("X", 100, 200.0, today - pd.Timedelta(days=340))
    # Position is underwater, so there is no gain to defer.
    adjusted = tax_aware_sell_priority(book, {"X": 10_000.0}, {"X": 100.0}, today)
    assert adjusted["X"] == pytest.approx(10_000.0)
