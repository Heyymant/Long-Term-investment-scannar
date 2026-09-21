//! User-uploaded portfolio (CSV / JSON) used as the live book.
//!
//! Market prints still come from NSE EOD rankings. This file only stores
//! *what you hold* (qty + avg cost) so BUY and SELL can both fire.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::data::artifacts::{column_f64, column_str};
use crate::data::models::{Holding, TableResponse};
use crate::signals::{classify, LiveName};

const BOOK_FILE: &str = "uploaded_book.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BookRowIn {
    #[serde(default, alias = "tradingsymbol", alias = "ticker", alias = "instrument")]
    pub symbol: String,
    #[serde(default)]
    pub isin: String,
    #[serde(default, alias = "quantity")]
    pub qty: f64,
    #[serde(default, alias = "average_price", alias = "avg")]
    pub avg_price: f64,
    #[serde(default, alias = "ltp", alias = "close")]
    pub last_price: f64,
    #[serde(default, alias = "exch")]
    pub exchange: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UploadedBook {
    /// True after a successful POST until DELETE. Distinguishes "no book"
    /// (unknown) from "cash-only book" (known empty holdings).
    #[serde(default)]
    pub present: bool,
    #[serde(default)]
    pub cash: f64,
    #[serde(default)]
    pub uploaded_at: String,
    #[serde(default)]
    pub rows: Vec<BookRowIn>,
}

impl UploadedBook {
    pub fn load(artifacts_dir: &Path) -> Self {
        let path = artifacts_dir.join(BOOK_FILE);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        serde_json::from_str(&text).unwrap_or_default()
    }

    pub fn save(&self, artifacts_dir: &Path) -> anyhow::Result<PathBuf> {
        let path = artifacts_dir.join(BOOK_FILE);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)?;
        Ok(path)
    }

    pub fn clear(artifacts_dir: &Path) {
        let path = artifacts_dir.join(BOOK_FILE);
        let _ = std::fs::remove_file(path);
    }

    pub fn names(&self) -> Vec<LiveName> {
        self.rows.iter().filter_map(row_to_live).collect()
    }
}

fn row_to_live(row: &BookRowIn) -> Option<LiveName> {
    let symbol = normalize_symbol(&row.symbol);
    let isin = row.isin.trim().to_uppercase();
    if symbol.is_empty() && isin.is_empty() {
        return None;
    }
    if row.qty <= 0.0 {
        return None;
    }
    let exchange = if row.exchange.trim().is_empty() {
        "NSE".into()
    } else {
        row.exchange.trim().to_uppercase()
    };
    Some(LiveName {
        symbol,
        isin,
        qty: row.qty,
        last_price: row.last_price.max(0.0),
        avg_price: row.avg_price.max(0.0),
        pnl: 0.0,
        exchange: exchange.clone(),
        asset_class: classify(&isin, &symbol, &exchange).into(),
    })
}

/// Strip exchange prefixes and series suffixes so "NSE:RELIANCE-EQ" → "RELIANCE".
pub fn normalize_symbol(raw: &str) -> String {
    let mut s = raw.trim().to_uppercase();
    if let Some((ex, rest)) = s.split_once(':') {
        if matches!(ex, "NSE" | "BSE" | "NFO" | "MCX" | "CDS" | "MF") {
            s = rest.trim().to_string();
        }
    }
    for suffix in ["-EQ", "-BE", "-BL", "-BZ", "-SM", "-ST", "-IL", ".NS", ".BO"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            s = stripped.to_string();
            break;
        }
    }
    s.trim().to_string()
}

pub fn parse_number(raw: &str) -> Option<f64> {
    let t = raw
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .replace(',', "")
        .replace('₹', "")
        .replace('%', "");
    if t.is_empty() || t == "-" || t == "—" || t.eq_ignore_ascii_case("na") {
        return None;
    }
    t.parse::<f64>().ok().filter(|x| x.is_finite())
}

/// Parse a broker CSV / TSV (Zerodha console, generic symbol,qty,avg).
pub fn parse_csv(text: &str) -> Result<(Vec<BookRowIn>, f64), String> {
    let text = text.trim_start_matches('\u{feff}').trim();
    if text.is_empty() {
        return Err("empty file".into());
    }
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#') && *l != "---")
        .collect();
    if lines.is_empty() {
        return Err("no rows".into());
    }

    let delim = detect_delim(lines[0]);
    let header = split_line(lines[0], delim);
    let header_keys: Vec<String> = header.iter().map(|h| canon_header(h)).collect();
    let has_header = header_keys.iter().any(|h| {
        matches!(
            h.as_str(),
            "symbol" | "qty" | "avg_price" | "last_price" | "isin" | "exchange"
        )
    });

    let mut cash = 0.0;
    let mut rows = Vec::new();
    let data_lines = if has_header { &lines[1..] } else { &lines[..] };
    for line in data_lines {
        let cols = split_line(line, delim);
        if cols.iter().all(|c| c.trim().is_empty()) {
            continue;
        }
        let get = |key: &str| -> String {
            if has_header {
                header_keys
                    .iter()
                    .position(|h| h == key)
                    .and_then(|i| cols.get(i))
                    .cloned()
                    .unwrap_or_default()
            } else {
                String::new()
            }
        };
        let symbol_raw = if has_header {
            get("symbol")
        } else {
            cols.first().cloned().unwrap_or_default()
        };
        let symbol = normalize_symbol(&symbol_raw);
        if symbol.is_empty() {
            continue;
        }
        let qty = if has_header {
            parse_number(&get("qty")).unwrap_or(0.0)
        } else {
            cols.get(1).and_then(|c| parse_number(c)).unwrap_or(1.0)
        };
        let avg_price = if has_header {
            parse_number(&get("avg_price")).unwrap_or(0.0)
        } else {
            cols.get(2).and_then(|c| parse_number(c)).unwrap_or(0.0)
        };
        let last_price = if has_header {
            parse_number(&get("last_price")).unwrap_or(0.0)
        } else {
            cols.get(3).and_then(|c| parse_number(c)).unwrap_or(0.0)
        };
        let isin = if has_header {
            get("isin").trim().to_uppercase()
        } else {
            String::new()
        };
        let exchange = if has_header {
            let e = get("exchange");
            if e.trim().is_empty() { "NSE".into() } else { e.trim().to_uppercase() }
        } else {
            "NSE".into()
        };
        if symbol == "CASH" || symbol == "LIQUID" {
            cash += if qty > 0.0 { qty } else { last_price.max(avg_price) };
            continue;
        }
        if qty <= 0.0 {
            continue;
        }
        rows.push(BookRowIn { symbol, isin, qty, avg_price, last_price, exchange });
    }
    if rows.is_empty() && cash <= 0.0 {
        return Err("no holdings found — need symbol and qty columns".into());
    }
    Ok((rows, cash))
}

fn detect_delim(header: &str) -> char {
    let commas = header.matches(',').count();
    let tabs = header.matches('\t').count();
    let semis = header.matches(';').count();
    if tabs > commas && tabs >= semis {
        '\t'
    } else if semis > commas {
        ';'
    } else {
        ','
    }
}

fn split_line(line: &str, delim: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
            continue;
        }
        if ch == delim && !in_quotes {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn canon_header(raw: &str) -> String {
    let h = raw
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .to_lowercase()
        .replace('.', "")
        .replace('_', " ")
        .replace('-', " ");
    let h = h.split_whitespace().collect::<Vec<_>>().join(" ");
    match h.as_str() {
        "symbol" | "tradingsymbol" | "ticker" | "instrument" | "nse symbol"
        | "scrip" | "stock" | "stock name" | "name" | "scrip name" | "trading symbol" => {
            "symbol".into()
        }
        "qty" | "quantity" | "qty available" | "shares" | "units" | "net qty"
        | "net quantity" | "qty." | "filled qty" => "qty".into(),
        "avg" | "avg price" | "average price" | "avg cost" | "average cost"
        | "buy price" | "avg. cost" | "average_price" => "avg_price".into(),
        "ltp" | "last" | "last price" | "close" | "price" | "last_price" | "cmp" => {
            "last_price".into()
        }
        "isin" => "isin".into(),
        "exchange" | "exch" | "ex" => "exchange".into(),
        other => other.replace(' ', "_"),
    }
}

/// Fill last price / ISIN / PnL from NSE EOD rankings. Never invents a print.
pub fn enrich_from_nse(live: &mut [LiveName], rankings: &TableResponse) {
    let symbols = column_str(rankings, "symbol");
    let isins = column_str(rankings, "isin");
    let last = column_f64(rankings, "last_price");
    let close = column_f64(rankings, "close");

    let mut by_sym: HashMap<String, (Option<String>, Option<f64>)> = HashMap::new();
    let mut by_isin: HashMap<String, (Option<String>, Option<f64>)> = HashMap::new();
    for i in 0..rankings.n_rows {
        let sym = symbols.get(i).cloned().flatten().map(|s| s.to_uppercase());
        let isin = isins.get(i).cloned().flatten().map(|s| s.to_uppercase());
        let px = last
            .get(i)
            .copied()
            .flatten()
            .filter(|p| *p > 0.0)
            .or_else(|| close.get(i).copied().flatten().filter(|p| *p > 0.0));
        if let Some(s) = &sym {
            by_sym.insert(s.clone(), (isin.clone(), px));
        }
        if let Some(id) = &isin {
            by_isin.insert(id.clone(), (sym.clone(), px));
        }
    }

    for name in live {
        let hit = by_sym
            .get(&name.symbol)
            .cloned()
            .or_else(|| {
                if name.isin.is_empty() {
                    None
                } else {
                    by_isin.get(&name.isin).cloned()
                }
            });
        if let Some((sym_or_isin, px)) = hit {
            if name.isin.is_empty() {
                if let Some(id) = sym_or_isin.filter(|s| s.starts_with("IN")) {
                    name.isin = id;
                }
            }
            if name.last_price <= 0.0 {
                if let Some(p) = px {
                    name.last_price = p;
                }
            }
        }
        if name.avg_price > 0.0 && name.last_price > 0.0 {
            name.pnl = (name.last_price - name.avg_price) * name.qty;
        }
    }
}

/// NSE snapshot for each uploaded (or Kite) line — used on the Book tab.
pub fn position_rows(live: &[LiveName], rankings: &TableResponse) -> Value {
    let idx: HashMap<&str, usize> = rankings
        .columns
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();
    let by_sym: HashMap<String, &Vec<Value>> = rankings
        .rows
        .iter()
        .filter_map(|r| {
            let key = idx.get("symbol").and_then(|&i| r.get(i)).and_then(Value::as_str)?;
            Some((key.to_uppercase(), r))
        })
        .collect();
    let by_isin: HashMap<String, &Vec<Value>> = rankings
        .rows
        .iter()
        .filter_map(|r| {
            let key = idx.get("isin").and_then(|&i| r.get(i)).and_then(Value::as_str)?;
            Some((key.to_uppercase(), r))
        })
        .collect();

    let mut missing = Vec::new();
    let mut matched = 0u32;
    let rows: Vec<Value> = live
        .iter()
        .map(|n| {
            let src = by_sym
                .get(&n.symbol)
                .copied()
                .or_else(|| by_isin.get(&n.isin).copied());
            let matched_nse = src.is_some();
            if matched_nse {
                matched += 1;
            } else {
                missing.push(n.symbol.clone());
            }
            let extra = |key: &str| -> Value {
                src.and_then(|r| idx.get(key).and_then(|&i| r.get(i)).cloned())
                    .unwrap_or(Value::Null)
            };
            json!({
                "symbol": n.symbol,
                "isin": n.isin,
                "qty": n.qty,
                "avg_price": n.avg_price,
                "last_price": n.last_price,
                "pnl": n.pnl,
                "value": n.value(),
                "exchange": n.exchange,
                "asset_class": n.asset_class,
                "matched_nse": matched_nse,
                "name": extra("name"),
                "sector": extra("sector"),
                "rank": extra("rank"),
                "pe": extra("pe"),
                "ret_1y": extra("ret_1y"),
                "prev_close": extra("prev_close"),
                "earnings_yield": extra("earnings_yield"),
                "net_margin": extra("net_margin"),
                "composite": extra("composite"),
            })
        })
        .collect();

    json!({
        "n": live.len(),
        "n_matched": matched,
        "missing": missing,
        "rows": rows,
    })
}

pub fn to_holdings(live: &[LiveName]) -> Vec<Holding> {
    live.iter()
        .map(|n| Holding {
            tradingsymbol: n.symbol.clone(),
            isin: n.isin.clone(),
            quantity: n.qty,
            average_price: n.avg_price,
            last_price: n.last_price,
            pnl: n.pnl,
            day_change_percentage: 0.0,
            exchange: n.exchange.clone(),
            instrument_token: 0,
            fund: String::new(),
        })
        .collect()
}

/// Equity (NSE/BSE) net positions from Kite — skip F&O.
pub fn positions_from_kite(raw: &Value) -> Vec<LiveName> {
    let mut out = Vec::new();
    let nets = raw
        .get("net")
        .and_then(Value::as_array)
        .or_else(|| raw.as_array());
    let Some(arr) = nets else { return out };
    for v in arr {
        if let Some(n) = live_from_position(v) {
            out.push(n);
        }
    }
    out
}

fn live_from_position(v: &Value) -> Option<LiveName> {
    let exchange = v
        .get("exchange")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_uppercase();
    if matches!(exchange.as_str(), "NFO" | "CDS" | "MCX" | "BCD") {
        return None;
    }
    let qty = json_num(v.get("quantity")).unwrap_or(0.0);
    if qty == 0.0 {
        return None;
    }
    let symbol = normalize_symbol(
        v.get("tradingsymbol")
            .and_then(Value::as_str)
            .unwrap_or(""),
    );
    let isin = v
        .get("isin")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_uppercase();
    if symbol.is_empty() && isin.is_empty() {
        return None;
    }
    Some(LiveName {
        symbol: symbol.clone(),
        isin: isin.clone(),
        qty,
        last_price: json_num(v.get("last_price")).unwrap_or(0.0),
        avg_price: json_num(v.get("average_price")).unwrap_or(0.0),
        pnl: json_num(v.get("pnl")).unwrap_or(0.0),
        exchange: if exchange.is_empty() { "NSE".into() } else { exchange.clone() },
        asset_class: classify(&isin, &symbol, &exchange).into(),
    })
}

/// Add a position only when the symbol/ISIN is not already on the book
/// (Kite holdings already cover CNC overnight).
pub fn merge_unique(base: &mut Vec<LiveName>, extra: Vec<LiveName>) {
    for n in extra {
        let exists = base.iter().any(|h| {
            (!n.isin.is_empty() && h.isin == n.isin)
                || (!n.symbol.is_empty() && h.symbol == n.symbol)
        });
        if !exists {
            base.push(n);
        }
    }
}

fn json_num(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_f64().or_else(|| x.as_i64().map(|n| n as f64)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_exchange_and_series() {
        assert_eq!(normalize_symbol("NSE:RELIANCE-EQ"), "RELIANCE");
        assert_eq!(normalize_symbol("infy.ns"), "INFY");
        assert_eq!(normalize_symbol("  TCS  "), "TCS");
    }

    #[test]
    fn parses_zerodha_console_csv() {
        let csv = "Instrument,Qty.,Avg. cost,LTP,Invested,Cur. val,P&L\n\
                   INFY,10,1500.00,1520.50,15000,15205,205\n\
                   RELIANCE,2,\"2,400.00\",2800,4800,5600,800\n";
        let (rows, cash) = parse_csv(csv).unwrap();
        assert_eq!(cash, 0.0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].symbol, "INFY");
        assert_eq!(rows[0].qty, 10.0);
        assert_eq!(rows[0].avg_price, 1500.0);
        assert_eq!(rows[1].symbol, "RELIANCE");
        assert_eq!(rows[1].avg_price, 2400.0);
    }

    #[test]
    fn cash_row_is_not_a_holding() {
        let csv = "symbol,qty,avg_price\nCASH,25000,1\nINFY,5,1400\n";
        let (rows, cash) = parse_csv(csv).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].symbol, "INFY");
        assert_eq!(cash, 25000.0);
    }

    #[test]
    fn headerless_symbol_qty_avg() {
        let csv = "TCS,15,3200\nHDFCBANK,8,1400\n";
        let (rows, _) = parse_csv(csv).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].symbol, "TCS");
        assert_eq!(rows[0].qty, 15.0);
        assert_eq!(rows[0].avg_price, 3200.0);
    }
}
