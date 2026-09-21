from src.data.asset_class import classify


def test_equity_isin():
    assert classify("INE002A01018", "RELIANCE", "NSE") == "equity"


def test_etf_isin_and_bees_symbol():
    assert classify("INF204KB14I2", "NIFTYBEES", "NSE") == "etf"
    assert classify("INE000000000", "GOLDBEES", "NSE") == "etf"


def test_mf_folio_channel():
    assert classify("INF090I01239", "HDFC_EQUITY", "MF") == "mf"
