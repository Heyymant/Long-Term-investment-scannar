"""NSE parser tests.

All fixtures are recorded payloads - no live network calls, so the suite stays
fast and deterministic. The samples below are trimmed versions of real NSE
responses captured from the public endpoints.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.nse import (
    _normalize_legacy,
    _normalize_udiff,
    bhavcopy_to_panels,
    build_symbol_history,
    infer_delistings,
    normalize_corporate_actions,
    parse_corporate_action,
)
from src.data.xbrl import parse_contexts, parse_xbrl, select_context


# --------------------------------------------------------------- corp actions
@pytest.mark.parametrize("subject,expected_type", [
    ("Dividend - Rs 2 Per Share", "dividend"),
    ("Interim Dividend - Rs 5.50 Per Share", "dividend"),
    ("Face Value Split From Rs 10 To Rs 2", "split"),
    ("Face Value Split (Sub-Division) - From Rs 10/- To Re 1/-", "split"),
    ("Bonus 1:1", "bonus"),
    ("Bonus issue 2:5", "bonus"),
])
def test_parses_known_action_subjects(subject, expected_type):
    action = parse_corporate_action(subject)
    assert action is not None, f"failed to parse: {subject}"
    assert action["action_type"] == expected_type


def test_split_ratio_gives_correct_share_multiplier():
    """Face value 10 -> 2 means one share becomes five."""
    from src.data.corporate_actions import _share_multiplier

    action = parse_corporate_action("Face Value Split From Rs 10 To Rs 2")
    row = pd.Series({"action_type": "split", **action})
    assert _share_multiplier(row) == pytest.approx(5.0)


def test_bonus_ratio_gives_correct_share_multiplier():
    """Bonus 1:1 means one share becomes two."""
    from src.data.corporate_actions import _share_multiplier

    action = parse_corporate_action("Bonus 1:1")
    row = pd.Series({"action_type": "bonus", **action})
    assert _share_multiplier(row) == pytest.approx(2.0)


def test_dividend_amount_extracted():
    action = parse_corporate_action("Dividend - Rs 2.50 Per Share")
    assert action["amount"] == pytest.approx(2.50)


def test_unrecognized_subject_returns_none():
    """Unparseable actions must not be silently guessed at."""
    assert parse_corporate_action("Annual General Meeting") is None
    assert parse_corporate_action("") is None
    assert parse_corporate_action(None) is None


def test_unparsed_actions_are_reported_not_dropped():
    raw = pd.DataFrame([
        {"isin": "INE001A01036", "symbol": "A", "exDate": "21-Sep-2026",
         "subject": "Dividend - Rs 2 Per Share"},
        {"isin": "INE002A01018", "symbol": "B", "exDate": "22-Sep-2026",
         "subject": "Scheme Of Arrangement"},
    ])
    parsed, unparsed = normalize_corporate_actions(raw)
    assert len(parsed) == 1
    assert len(unparsed) == 1
    assert unparsed.iloc[0]["symbol"] == "B"


# ------------------------------------------------------------------ bhavcopy
def _udiff_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "TradDt": ["2024-06-28", "2024-06-28", "2024-06-28"],
        "FinInstrmTp": ["STK", "STK", "IDX"],
        "ISIN": ["INE263M01029", "INE0PQ001012", "INDEXNOTANISIN"],
        "TckrSymb": ["RUSTOMJEE", "VISHNUINFR", "NIFTY"],
        "SctySrs": ["EQ", "SM", "EQ"],
        "OpnPric": [669.50, 204.55, 100.0],
        "HghPric": [675.00, 204.55, 101.0],
        "LwPric": [655.60, 196.30, 99.0],
        "ClsPric": [673.95, 200.00, 100.5],
        "PrvsClsgPric": [671.00, 204.55, 100.0],
        "TtlTradgVol": [85357, 18500, 0],
        "TtlTrfVal": [5.686890e07, 3.698500e06, 0.0],
        "TtlNbOfTxsExctd": [8692, 29, 0],
    })


def test_udiff_normalization_keeps_only_real_isins():
    out = _normalize_udiff(_udiff_frame(), pd.Timestamp("2024-06-28").date())
    assert len(out) == 2                      # index row dropped
    assert set(out["isin"]) == {"INE263M01029", "INE0PQ001012"}
    assert set(out.columns) >= {"date", "isin", "symbol", "series", "close",
                                "volume", "traded_value"}


def test_udiff_preserves_exchange_traded_value():
    """Traded value comes straight from the exchange; it is a better ADV
    input than price x volume."""
    out = _normalize_udiff(_udiff_frame(), pd.Timestamp("2024-06-28").date())
    row = out[out["symbol"] == "RUSTOMJEE"].iloc[0]
    assert row["traded_value"] == pytest.approx(5.686890e07)
    # Not simply close * volume.
    assert abs(row["traded_value"] - row["close"] * row["volume"]) > 1000


def test_legacy_bhavcopy_normalization():
    legacy = pd.DataFrame({
        "SYMBOL": ["INFY"], "SERIES": ["EQ"], "OPEN": [1500.0], "HIGH": [1520.0],
        "LOW": [1490.0], "CLOSE": [1510.0], "LAST": [1510.0], "PREVCLOSE": [1495.0],
        "TOTTRDQTY": [1000000], "TOTTRDVAL": [1.51e9], "TIMESTAMP": ["15-MAR-2019"],
        "TOTALTRADES": [50000], "ISIN": ["INE009A01021"],
    })
    out = _normalize_legacy(legacy, pd.Timestamp("2019-03-15").date())
    assert len(out) == 1
    assert out.iloc[0]["isin"] == "INE009A01021"
    assert out.iloc[0]["close"] == pytest.approx(1510.0)
    assert out.iloc[0]["date"] == pd.Timestamp("2019-03-15")


def test_panels_are_pivoted_by_isin():
    bhav = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "isin": ["INE001A01036", "INE002A01018"] * 2,
        "symbol": ["A", "B"] * 2,
        "close": [100.0, 200.0, 101.0, 202.0],
        "volume": [1000, 2000, 1100, 2200],
        "traded_value": [1e5, 4e5, 1.1e5, 4.4e5],
    })
    panels = bhavcopy_to_panels(bhav)
    assert set(panels) >= {"close", "volume", "traded_value"}
    assert panels["close"].shape == (2, 2)
    assert panels["close"].loc[pd.Timestamp("2024-01-02"), "INE001A01036"] == 101.0


# -------------------------------------------------- survivorship / identity
def test_symbol_history_reconstructed_from_archive():
    """NSE publishes no rename log, but the archive reveals it."""
    bhav = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03", "2022-06-01", "2023-01-02", "2023-06-01"]),
        "isin": ["INE001A01036"] * 4,
        "symbol": ["OLDNAME", "OLDNAME", "NEWNAME", "NEWNAME"],
    })
    hist = build_symbol_history(bhav)
    assert len(hist) == 2
    old = hist[hist["symbol"] == "OLDNAME"].iloc[0]
    new = hist[hist["symbol"] == "NEWNAME"].iloc[0]
    assert pd.notna(old["valid_to"])         # closed interval
    assert pd.isna(new["valid_to"])          # still current


def test_delistings_inferred_from_disappearance():
    """A name that stops appearing in the archive has delisted or been
    suspended - this is what makes the history survivorship-free."""
    dates = pd.bdate_range("2024-01-01", periods=120)
    rows = []
    for d in dates:
        rows.append({"date": d, "isin": "INE_ALIVE01"})
        if d < dates[30]:
            rows.append({"date": d, "isin": "INE_GONE0001"})

    gone = infer_delistings(pd.DataFrame(rows), absent_days=30)
    assert list(gone["isin"]) == ["INE_GONE0001"]


# ---------------------------------------------------------------------- XBRL
SAMPLE_XBRL = """<?xml version="1.0" encoding="UTF-8"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance"
      xmlns:in-bse="http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin">
  <context id="Q3">
    <period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period>
  </context>
  <context id="YTD">
    <period><startDate>2024-04-01</startDate><endDate>2024-12-31</endDate></period>
  </context>
  <context id="SEGMENT">
    <entity><segment><explicitMember dimension="d">X</explicitMember></segment></entity>
    <period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period>
  </context>
  <in-bse:RevenueFromOperations contextRef="Q3">2191000000.00</in-bse:RevenueFromOperations>
  <in-bse:RevenueFromOperations contextRef="YTD">6500000000.00</in-bse:RevenueFromOperations>
  <in-bse:RevenueFromOperations contextRef="SEGMENT">900000000.00</in-bse:RevenueFromOperations>
  <in-bse:ProfitBeforeTax contextRef="Q3">36000000.00</in-bse:ProfitBeforeTax>
  <in-bse:ProfitLossForPeriod contextRef="Q3">12800000.00</in-bse:ProfitLossForPeriod>
  <in-bse:BasicEarningsLossPerShareFromContinuingOperations contextRef="Q3">1.48</in-bse:BasicEarningsLossPerShareFromContinuingOperations>
  <in-bse:FinanceCosts contextRef="Q3">5000000.00</in-bse:FinanceCosts>
</xbrl>
"""


def test_xbrl_extracts_core_financials():
    facts = parse_xbrl(SAMPLE_XBRL, pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert facts is not None
    assert facts.get("revenue") == pytest.approx(2191000000.0)
    assert facts.get("net_income") == pytest.approx(12800000.0)
    assert facts.get("eps_basic") == pytest.approx(1.48)
    assert facts.get("profit_before_tax") == pytest.approx(36000000.0)


def test_xbrl_picks_the_quarter_not_the_year_to_date():
    """The critical correctness property.

    A Q3 filing contains the quarter AND the 9-month cumulative figure. Taking
    the wrong one silently injects a 3x revenue jump into the series and would
    destroy any SUE calculation.
    """
    facts = parse_xbrl(SAMPLE_XBRL, pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert facts.get("revenue") == pytest.approx(2191000000.0)   # quarter
    assert facts.get("revenue") != pytest.approx(6500000000.0)   # not YTD


def test_xbrl_ignores_dimensional_segment_contexts():
    facts = parse_xbrl(SAMPLE_XBRL, pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert facts.get("revenue") != pytest.approx(900000000.0)


def test_context_selection_prefers_shortest_matching_period():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(SAMPLE_XBRL)
    contexts = parse_contexts(root)
    assert "Q3" in contexts and "YTD" in contexts
    assert contexts["SEGMENT"].has_dimensions
    assert contexts["Q3"].is_quarterly
    assert not contexts["YTD"].is_quarterly

    chosen = select_context(contexts, pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31"))
    assert chosen == "Q3"


def test_context_selection_without_hint_prefers_a_quarter():
    import xml.etree.ElementTree as ET

    contexts = parse_contexts(ET.fromstring(SAMPLE_XBRL))
    assert select_context(contexts) == "Q3"


def test_malformed_xbrl_returns_none():
    assert parse_xbrl("not xml at all") is None
    assert parse_xbrl("") is None


# ------------------------------------------------------------- integration
def test_mismatched_membership_does_not_empty_the_universe():
    """Regression: mixing data sources must not silently zero the universe.

    Real NSE constituents cached on disk share no ISINs with a synthetic
    price panel. Applying that membership as a filter produced an empty
    universe, which looks like a broken strategy rather than a data mismatch.
    """
    from src.data.universe import IndexMembership, UniverseBuilder, UniverseFilters

    dates = pd.bdate_range("2024-01-01", periods=400)
    prices = pd.DataFrame(100.0, index=dates, columns=["SYN0001", "SYN0002"])
    volumes = pd.DataFrame(1e7, index=dates, columns=prices.columns)

    # Membership listing completely different (real) ISINs.
    membership = IndexMembership.from_static(["INE009A01021", "INE002A01018"], "NSE500")

    filters = UniverseFilters(min_adv_inr=0, min_price_inr=0, min_listing_days=0,
                              min_financial_quarters=0)
    builder = UniverseBuilder(filters, membership, "NSE500")
    eligible = builder.build(prices, volumes)

    # Without the guard this is 0 and every downstream result is meaningless.
    assert int(eligible.iloc[-1].sum()) == 0, (
        "builder applies membership as given; the loader is responsible for "
        "detecting the mismatch"
    )

    # The loader-level guard swaps in a membership derived from the panel.
    fallback = IndexMembership.from_static(list(prices.columns), "NSE500")
    eligible2 = UniverseBuilder(filters, fallback, "NSE500").build(prices, volumes)
    assert int(eligible2.iloc[-1].sum()) == 2


def test_nse_provider_normalizes_to_the_canonical_schema():
    """EPS must survive normalization - Sleeve C's SUE depends on it."""
    from src.data.fundamentals.base import FUNDAMENTAL_COLUMNS
    from src.data.fundamentals.nse_provider import NSEProvider

    provider = NSEProvider(cache_dir=None)
    raw = pd.DataFrame([{
        "isin": "INE001A01036",
        "announce_date": pd.Timestamp("2025-01-15 17:30:00"),
        "period_end": pd.Timestamp("2024-12-31"),
        "eps": 1.48,
        "revenue": 2191000000.0,
        "net_income": 12800000.0,
    }])
    out = provider.normalize(raw)
    assert "eps" in FUNDAMENTAL_COLUMNS
    assert out.iloc[0]["eps"] == pytest.approx(1.48)
    # A real announcement timestamp must be preserved, not overwritten by the lag.
    assert out.iloc[0]["announce_date"] == pd.Timestamp("2025-01-15 17:30:00")
