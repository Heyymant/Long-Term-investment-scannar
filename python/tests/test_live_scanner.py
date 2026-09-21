"""Price-movement scanner: latch, threshold, token directory."""

from datetime import datetime, timezone

from src.signals.live_scanner import PriceMovementScanner


def _scanner() -> PriceMovementScanner:
    s = PriceMovementScanner(pct_threshold=0.03)
    s.register("ZENTEC", 100.0, token=101)
    s.register("INFY", 1500.0, token=408065)
    return s


def test_first_cross_emits_and_latches():
    s = _scanner()
    alert = s.scan(symbol="ZENTEC", last=104.0)  # +4%
    assert alert is not None
    assert alert.symbol == "ZENTEC"
    assert abs(alert.abs_return - 0.04) < 1e-9
    assert "moved by 4.00%" in alert.format()
    assert s.scan(symbol="ZENTEC", last=110.0) is None  # latched


def test_below_threshold_is_silent():
    s = _scanner()
    assert s.scan(symbol="INFY", last=1540.0) is None  # +2.67%


def test_token_directory_resolves_symbol():
    s = _scanner()
    ts = datetime(2026, 4, 24, 4, 0, 0, tzinfo=timezone.utc)
    alert = s.scan(token=408065, last=1600.0, ts=ts)  # +6.67%
    assert alert is not None
    assert alert.symbol == "INFY"
    assert alert.ts == ts


def test_reset_latches_allows_a_second_alert():
    s = _scanner()
    assert s.scan(symbol="ZENTEC", last=104.0) is not None
    s.reset_latches()
    assert s.scan(symbol="ZENTEC", last=104.0) is not None


def test_unknown_or_bad_prices_are_ignored():
    s = _scanner()
    assert s.scan(symbol="UNKNOWN", last=10.0) is None
    assert s.scan(symbol="ZENTEC", last=0.0) is None
    assert s.scan(token=999) is None
