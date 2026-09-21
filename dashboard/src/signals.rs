//! BUY / SELL / HOLD / WATCH from the rank-buffer trading rules.
//!
//! Python writes `trade_signals.parquet` on a new backtest. For older runs
//! (or if that file is missing) we rebuild the list from rankings + target
//! holdings + strategy_config so the Signals tab is never empty.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use crate::data::artifacts::{column_f64, column_str};
use crate::data::models::{Holding, TableResponse};

const MIN_ORDER_VALUE: f64 = 5_000.0;
const ETF_TOP_N: usize = 8;
const ETF_TOKENS: &[&str] = &[
    "ETF", "BEES", "IETF", "SETFNIF", "SETFNN50", "MON100", "MOSNIFTY",
    "NIFTYBEES", "GOLDBEES", "BANKBEES", "LIQUIDBE", "JUNIORBE",
];

pub fn classify(isin: &str, symbol: &str, exchange: &str) -> &'static str {
    let isin_u = isin.trim().to_uppercase();
    let sym = symbol.trim().to_uppercase();
    let ex = exchange.trim().to_uppercase();
    if ex == "MF" || ex == "MFSS" {
        return "mf";
    }
    let etf_name = ETF_TOKENS.iter().any(|t| sym.contains(t));
    if isin_u.starts_with("INF") || etf_name {
        return "etf";
    }
    "equity"
}

/// One line from the live Kite book (or a backtest proxy).
#[derive(Debug, Clone, Default)]
pub struct LiveName {
    pub symbol: String,
    pub isin: String,
    pub qty: f64,
    pub last_price: f64,
    pub avg_price: f64,
    pub pnl: f64,
    pub exchange: String,
    pub asset_class: String,
}

impl LiveName {
    pub fn from_holding(h: &Holding) -> Self {
        let symbol = if h.tradingsymbol.trim().is_empty() {
            h.fund.trim().to_uppercase()
        } else {
            h.tradingsymbol.trim().to_uppercase()
        };
        let isin = h.isin.trim().to_uppercase();
        let exchange = h.exchange.trim().to_uppercase();
        let asset_class = classify(&isin, &symbol, &exchange).to_string();
        Self {
            symbol, isin, qty: h.quantity, last_price: h.last_price,
            avg_price: h.average_price, pnl: h.pnl, exchange, asset_class,
        }
    }

    pub fn from_mf_json(v: &Value) -> Option<Self> {
        let qty = v.get("quantity").and_then(json_f64).unwrap_or(0.0);
        if qty <= 0.0 {
            return None;
        }
        let symbol = v.get("tradingsymbol")
            .or_else(|| v.get("fund"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_uppercase();
        let isin = v.get("isin").and_then(Value::as_str).unwrap_or("").trim().to_uppercase();
        if symbol.is_empty() && isin.is_empty() {
            return None;
        }
        Some(Self {
            symbol,
            isin,
            qty,
            last_price: v.get("last_price").and_then(json_f64).unwrap_or(0.0),
            avg_price: v.get("average_price").and_then(json_f64).unwrap_or(0.0),
            pnl: v.get("pnl").and_then(json_f64).unwrap_or(0.0),
            exchange: "MF".into(),
            asset_class: "mf".into(),
        })
    }

    pub fn value(&self) -> f64 {
        if self.last_price > 0.0 { self.qty * self.last_price } else { 0.0 }
    }
}

pub struct SignalsPayload {
    pub summary: Value,
    pub signals: TableResponse,
    pub buys: TableResponse,
    pub sells: TableResponse,
    pub queued: TableResponse,
    pub watch: TableResponse,
    pub holds: TableResponse,
    pub screener: TableResponse,
}

impl SignalsPayload {
    pub fn into_json(self) -> Value {
        json!({
            "summary": self.summary,
            "signals": self.signals,
            "buys": self.buys,
            "sells": self.sells,
            "queued": self.queued,
            "watch": self.watch,
            "holds": self.holds,
            "screener": self.screener,
            "note": "Decision support only — no orders are placed. Execute manually.",
        })
    }
}

pub fn from_artifact(table: TableResponse, mut summary: Value) -> SignalsPayload {
    let buys = filter_action(&table, "BUY");
    let sells = filter_action(&table, "SELL");
    let watch = filter_action(&table, "WATCH");
    let holds = filter_action(&table, "HOLD");
    let top_n = summary.get("top_n").and_then(Value::as_u64).unwrap_or(40) as f64;
    let queued = filter_queued(&table, top_n);
    if summary.get("n_queued").is_none() {
        if let Some(obj) = summary.as_object_mut() {
            obj.insert("n_queued".into(), json!(queued.n_rows));
        }
    }
    SignalsPayload { summary, signals: table.clone(), buys, sells, queued, watch, holds, screener: table }
}

/// Rebuild from rankings + a holdings table (backtest targets, no live qty).
pub fn from_rankings(
    rankings: &TableResponse,
    holdings: &TableResponse,
    config: &Value,
) -> SignalsPayload {
    suggest(
        rankings, &live_from_table(holdings), &HashMap::new(), config, 0.0,
        "backtest_target", "not connected", &HashMap::new(),
    )
}

/// Yesterday's close + token from a Kite quote snapshot (Databento last_day_lookup).
#[derive(Debug, Clone, Default)]
pub struct QuoteSnap {
    pub last: f64,
    pub prev_close: Option<f64>,
    pub token: Option<u32>,
    pub volume: Option<f64>,
}

/// Rank-buffer screener sized against a live Kite book.
pub fn suggest(
    rankings: &TableResponse,
    live: &[LiveName],
    quotes: &HashMap<String, f64>,
    config: &Value,
    cash: f64,
    book_source: &str,
    kite_status: &str,
    snaps: &HashMap<String, QuoteSnap>,
) -> SignalsPayload {
    let top_n = config.pointer("/sleeve_a/top_n").and_then(Value::as_u64).unwrap_or(40) as usize;
    let buffer_mult = config.pointer("/sleeve_a/rank_buffer_multiple").and_then(Value::as_f64).unwrap_or(2.0);
    let mom_w = config.pointer("/sleeve_a/factors/weight_momentum").and_then(Value::as_f64).unwrap_or(0.0);
    let q_gate = config.pointer("/sleeve_a/factors/value_quality_gate").and_then(Value::as_bool).unwrap_or(true);
    let q_w = config.pointer("/sleeve_a/factors/weight_quality").and_then(Value::as_f64).unwrap_or(0.0);
    let lv_w = config.pointer("/sleeve_a/factors/weight_low_vol").and_then(Value::as_f64).unwrap_or(0.0);
    let max_w = config.pointer("/sleeve_a/max_weight_per_stock").and_then(Value::as_f64).unwrap_or(0.05);

    let n_scored = rankings.n_rows.max(1);
    let buffer_n = ((top_n as f64) * buffer_mult.max(1.0)).round() as usize;
    let buffer_n = buffer_n.min(n_scored).max(top_n);
    let etf_top = ETF_TOP_N;
    let etf_buffer = ((etf_top as f64) * buffer_mult.max(1.0)).round() as usize;

    let book = index_live(live);
    let equity: f64 = live.iter().map(LiveName::value).sum();
    let nav = (equity + cash.max(0.0)).max(0.0);

    let isins = column_str(rankings, "isin");
    let symbols = column_str(rankings, "symbol");
    let sectors = column_str(rankings, "sector");
    let classes = column_str(rankings, "asset_class");
    let ranks = column_f64(rankings, "rank");
    let composites = column_f64(rankings, "composite");
    let qualities = column_f64(rankings, "quality");
    let values = column_f64(rankings, "value");
    let momentums = column_f64(rankings, "momentum");
    let low_vols = column_f64(rankings, "low_vol");
    let abs_moms = column_f64(rankings, "abs_momentum");
    let profits = column_f64(rankings, "profitability");

    let has_mom = momentums.iter().any(Option::is_some);
    let has_q = qualities.iter().any(Option::is_some);
    let has_v = values.iter().any(Option::is_some);
    let has_lv = low_vols.iter().any(Option::is_some);
    let mom_on = mom_w > 0.0 && has_mom;
    let gate_on = q_gate && has_q && has_v;
    let quality_on = q_w > 0.0 && has_q;
    let lv_on = lv_w > 0.0 && has_lv;

    let mut keep = 0usize;
    let mut etf_keep = 0usize;
    let mut matched: HashSet<String> = HashSet::new();
    for i in 0..rankings.n_rows {
        let isin = isins.get(i).cloned().flatten().unwrap_or_default();
        let symbol = symbols.get(i).cloned().flatten().unwrap_or_default();
        let rank = ranks.get(i).copied().flatten().unwrap_or(f64::MAX);
        let asset = classes.get(i).cloned().flatten().unwrap_or_else(|| classify(&isin, &symbol, "").into());
        if lookup(&book, &isin, &symbol).is_some() {
            matched.insert(key(&isin, &symbol));
            if asset == "etf" && rank <= etf_buffer as f64 {
                etf_keep += 1;
            } else if asset == "equity" && rank <= buffer_n as f64 {
                keep += 1;
            }
        }
    }
    let room = top_n.saturating_sub(keep);
    let etf_room = etf_top.saturating_sub(etf_keep);
    let mut buy_used = 0usize;
    let mut etf_buy_used = 0usize;

    let mut columns = suggestion_columns();
    let mut rows: Vec<Vec<Value>> = Vec::new();
    let mut selected: Vec<(String, f64)> = Vec::new(); // (key, composite-for-tilt)

    let mut order: Vec<usize> = (0..rankings.n_rows).collect();
    order.sort_by(|&a, &b| {
        let ra = ranks.get(a).copied().flatten().unwrap_or(f64::MAX);
        let rb = ranks.get(b).copied().flatten().unwrap_or(f64::MAX);
        ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in order {
        let isin = isins.get(i).cloned().flatten().unwrap_or_default();
        if isin.is_empty() && symbols.get(i).cloned().flatten().unwrap_or_default().is_empty() {
            continue;
        }
        let symbol = symbols.get(i).cloned().flatten().unwrap_or_else(|| isin.clone());
        let rank = ranks.get(i).copied().flatten().unwrap_or(f64::MAX);
        let live_row = lookup(&book, &isin, &symbol);
        let is_held = live_row.is_some();
        let sector = sectors.get(i).cloned().flatten().unwrap_or_else(|| "UNKNOWN".into());
        let composite = composites.get(i).copied().flatten();
        let quality = qualities.get(i).copied().flatten();
        let value = values.get(i).copied().flatten();
        let momentum = momentums.get(i).copied().flatten();
        let low_vol = low_vols.get(i).copied().flatten();
        let abs_mom = abs_moms.get(i).copied().flatten();
        let profitability = profits.get(i).copied().flatten();
        let asset = classes.get(i).cloned().flatten()
            .unwrap_or_else(|| classify(&isin, &symbol, live_row.map(|n| n.exchange.as_str()).unwrap_or("")).into());
        let is_etf = asset == "etf";
        let use_top = if is_etf { etf_top } else { top_n };
        let use_buf = if is_etf { etf_buffer } else { buffer_n };
        let use_room = if is_etf { etf_room } else { room };
        let buy_slot = if is_etf { &mut etf_buy_used } else { &mut buy_used };
        let sleeve = if is_etf { "ETF" } else { "A" };

        let book_loaded = book_source != "empty";
        let (action_opt, rules, reason) = decide(
            is_held, rank, use_top, use_buf, use_room, buy_slot,
            momentum, quality, value, low_vol, abs_mom, profitability, mom_on,
            gate_on && !is_etf, quality_on && !is_etf, lv_on, is_etf, &asset,
            book_loaded,
        );
        let action = action_opt.unwrap_or_else(|| "PASS".into());
        if (action == "BUY" || action == "HOLD") && asset != "mf" {
            selected.push((key(&isin, &symbol), composite.unwrap_or(0.0).max(0.1)));
        }

        let conv = conviction(&action, composite, rank, use_top, use_buf);
        let pos = size_row(live_row, &symbol, quotes, nav, 0.0);
        rows.push(row_values(
            &symbol, &isin, &action, sleeve, rank, composite, quality, value, momentum, low_vol,
            &sector, is_held, pos, conv, &rules, &reason, &asset,
        ));
    }

    // Holdings that the research universe does not cover.
    for name in live {
        let k = key(&name.isin, &name.symbol);
        if matched.contains(&k) {
            continue;
        }
        let pos = size_row(Some(name), &name.symbol, quotes, nav, 0.0);
        let (action, rules, reason) = match name.asset_class.as_str() {
            "mf" => (
                "HOLD",
                "mf_satellite",
                "Mutual-fund folio: keep as a satellite. Prefer a listed ETF when adding.",
            ),
            "etf" => (
                "WATCH",
                "etf_unscored",
                "Listed ETF is not in the liquid momentum/low-vol sleeve yet. Review; do not auto-sell.",
            ),
            _ => (
                "SELL",
                "outside_universe",
                "Not in the current research universe (filters, listing age, or missing scores). Review and sell if it no longer belongs.",
            ),
        };
        let sleeve = if name.asset_class == "mf" { "MF" } else if name.asset_class == "etf" { "ETF" } else { "A" };
        rows.push(row_values(
            &name.symbol, &name.isin, action, sleeve, f64::NAN, None, None, None, None, None,
            "UNKNOWN", true, pos, 0.5, rules, reason, &name.asset_class,
        ));
    }

    let targets = target_weights(&selected, max_w);
    apply_targets(&mut rows, &targets, quotes, nav);

    rows.sort_by(|a, b| {
        let oa = action_ord(a[2].as_str().unwrap_or(""));
        let ob = action_ord(b[2].as_str().unwrap_or(""));
        oa.cmp(&ob).then_with(|| {
            let ca = json_f64(&a[20]).unwrap_or(0.0);
            let cb = json_f64(&b[20]).unwrap_or(0.0);
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let eod_px = ranking_eod_map(rankings);
    for row in &mut rows {
        let symbol = row.first().and_then(Value::as_str).unwrap_or("").to_string();
        let mut last = json_f64(row.get(13).unwrap_or(&Value::Null)).unwrap_or(0.0);
        if last <= 0.0 {
            last = eod_px.get(&symbol.to_uppercase()).and_then(|e| e.last).unwrap_or(0.0);
        }
        let eod = eod_px.get(&symbol.to_uppercase());
        push_live(row, &symbol, snaps, last, eod.and_then(|e| e.prev), eod.and_then(|e| e.volume));
    }
    attach_screener_extras(&mut rows, &mut columns, rankings);
    fill_eod_change(&mut rows, &columns);

    let screener = TableResponse { n_rows: rows.len(), columns: columns.clone(), rows: rows.clone(), note: None };
    let actionable: Vec<Vec<Value>> = rows
        .into_iter()
        .filter(|r| matches!(r[2].as_str(), Some("BUY" | "SELL" | "WATCH" | "HOLD")))
        .collect();
    let table = TableResponse { n_rows: actionable.len(), columns, rows: actionable, note: None };
    let buy_notional: f64 = sum_delta(&table, "BUY");
    let sell_notional: f64 = sum_delta(&table, "SELL").abs();

    let mut payload = from_artifact(table, json!({
        "as_of_ranks": n_scored,
        "top_n": top_n,
        "buffer_n": buffer_n,
        "n_keep": keep,
        "n_room": room,
        "etf_top_n": etf_top,
        "etf_buffer_n": etf_buffer,
        "n_etf_keep": etf_keep,
        "n_etf_room": etf_room,
        "equity_value": equity,
        "cash": cash,
        "nav": nav,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "max_weight_per_stock": max_w,
        "min_order_value": MIN_ORDER_VALUE,
        "book_source": book_source,
        "kite_status": kite_status,
        "login_url": "/kite/login",
        "rules": rule_copy(top_n, buffer_n, mom_on, gate_on),
        "note": "Decision support only — no orders are placed. Execute manually.",
        "source": "nse_screener",
    }));
    if let Some(obj) = payload.summary.as_object_mut() {
        obj.insert("n_held".into(), json!(live.len()));
        obj.insert("n_keep".into(), json!(keep));
        obj.insert("n_room".into(), json!(room));
        obj.insert("etf_top_n".into(), json!(etf_top));
        obj.insert("n_etf_keep".into(), json!(etf_keep));
        obj.insert("n_etf_room".into(), json!(etf_room));
        obj.insert("n_equity_held".into(), json!(live.iter().filter(|n| n.asset_class == "equity").count()));
        obj.insert("n_etf_held".into(), json!(live.iter().filter(|n| n.asset_class == "etf").count()));
        obj.insert("n_mf_held".into(), json!(live.iter().filter(|n| n.asset_class == "mf").count()));
        obj.insert("equity_value".into(), json!(equity));
        obj.insert("cash".into(), json!(cash));
        obj.insert("nav".into(), json!(nav));
        obj.insert("buy_notional".into(), json!(buy_notional));
        obj.insert("sell_notional".into(), json!(sell_notional));
        obj.insert("book_source".into(), json!(book_source));
        obj.insert("kite_status".into(), json!(kite_status));
        obj.insert("login_url".into(), json!("/kite/login"));
        obj.insert("source".into(), json!("nse_screener"));
        obj.insert("n_buys".into(), json!(payload.buys.n_rows));
        obj.insert("n_sells".into(), json!(payload.sells.n_rows));
        obj.insert("n_holds".into(), json!(payload.holds.n_rows));
        obj.insert("n_watch".into(), json!(payload.watch.n_rows));
        obj.insert("n_queued".into(), json!(payload.queued.n_rows));
        obj.insert("scanner_threshold".into(), json!(0.03));
        obj.insert("n_quoted".into(), json!(snaps.len()));
        let dir: Vec<Value> = snaps.iter().filter_map(|(sym, s)| {
            s.token.map(|t| json!({ "symbol": sym, "token": t, "prev_close": s.prev_close }))
        }).collect();
        obj.insert("directory".into(), json!(dir));
    }
    payload.screener = screener;
    payload
}

struct EodPx {
    last: Option<f64>,
    prev: Option<f64>,
    volume: Option<f64>,
}

fn ranking_eod_map(rankings: &TableResponse) -> HashMap<String, EodPx> {
    let symbols = column_str(rankings, "symbol");
    let last = column_f64(rankings, "last_price");
    let close = column_f64(rankings, "close");
    let prev = column_f64(rankings, "prev_close");
    let vol = column_f64(rankings, "volume");
    let mut out = HashMap::new();
    for i in 0..rankings.n_rows {
        let Some(sym) = symbols.get(i).cloned().flatten() else { continue };
        let px = last.get(i).copied().flatten().filter(|p| *p > 0.0)
            .or_else(|| close.get(i).copied().flatten().filter(|p| *p > 0.0));
        out.insert(sym.to_uppercase(), EodPx {
            last: px,
            prev: prev.get(i).copied().flatten().filter(|p| *p > 0.0),
            volume: vol.get(i).copied().flatten().filter(|p| *p > 0.0),
        });
    }
    out
}

fn missing_num(v: &Value) -> bool {
    match json_f64(v) {
        None => true,
        Some(x) => !x.is_finite(),
    }
}

fn missing_price(v: &Value) -> bool {
    json_f64(v).map(|x| !(x > 0.0)).unwrap_or(true)
}

fn attach_screener_extras(
    rows: &mut [Vec<Value>],
    columns: &mut Vec<String>,
    rankings: &TableResponse,
) {
    let extras = [
        "name", "exchange", "country", "industry",
        "close", "last_price", "prev_close", "volume", "day_chg",
        "pe", "pb", "market_cap", "roe", "debt_equity", "div_yield", "payout",
        "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "ret_3y",
        "revenue", "eps", "net_income", "earnings_yield", "eps_ttm", "net_margin",
        "pretax_margin", "gross_profitability",
        "profitability", "growth", "safety", "payout_z",
    ];
    let rsym = rankings.columns.iter().position(|c| c == "symbol");
    let risin = rankings.columns.iter().position(|c| c == "isin");
    let rclose = rankings.columns.iter().position(|c| c == "close");
    let rlast = rankings.columns.iter().position(|c| c == "last_price");
    let by_sym: HashMap<String, &Vec<Value>> = rankings
        .rows
        .iter()
        .filter_map(|r| {
            let key = rsym.and_then(|i| r.get(i)).and_then(Value::as_str)?;
            Some((key.to_uppercase(), r))
        })
        .collect();
    let by_isin: HashMap<String, &Vec<Value>> = rankings
        .rows
        .iter()
        .filter_map(|r| {
            let key = risin.and_then(|i| r.get(i)).and_then(Value::as_str)?;
            Some((key.to_uppercase(), r))
        })
        .collect();
    let isin_i = columns.iter().position(|c| c == "isin");
    let sector_i = columns.iter().position(|c| c == "sector");

    for extra in extras {
        let src = rankings.columns.iter().position(|c| c == extra);
        let existing = columns.iter().position(|c| c == extra);
        if let Some(dst) = existing {
            for row in rows.iter_mut() {
                let fill_px = matches!(extra, "last_price" | "close" | "prev_close" | "volume")
                    && missing_price(row.get(dst).unwrap_or(&Value::Null));
                let fill_num = !matches!(extra, "last_price" | "close" | "prev_close" | "volume")
                    && missing_num(row.get(dst).unwrap_or(&Value::Null));
                if !(fill_px || fill_num) {
                    continue;
                }
                let sym = row.first().and_then(Value::as_str).unwrap_or("").to_uppercase();
                let isin = isin_i.and_then(|i| row.get(i)).and_then(Value::as_str).unwrap_or("").to_uppercase();
                let src_row = by_sym.get(&sym).copied().or_else(|| by_isin.get(&isin).copied());
                let mut val = src.and_then(|i| src_row.and_then(|r| r.get(i))).cloned();
                if extra == "last_price" && val.as_ref().map(missing_price).unwrap_or(true) {
                    val = rlast.and_then(|i| src_row.and_then(|r| r.get(i))).cloned()
                        .filter(|v| !missing_price(v))
                        .or_else(|| rclose.and_then(|i| src_row.and_then(|r| r.get(i))).cloned());
                }
                if let Some(v) = val {
                    if extra == "last_price" || extra == "close" || extra == "prev_close" || extra == "volume" {
                        if !missing_price(&v) {
                            row[dst] = v;
                        }
                    } else if !missing_num(&v) {
                        row[dst] = v;
                    }
                }
            }
            continue;
        }
        columns.push(extra.to_string());
        for row in rows.iter_mut() {
            let sym = row.first().and_then(Value::as_str).unwrap_or("").to_uppercase();
            let isin = isin_i.and_then(|i| row.get(i)).and_then(Value::as_str).unwrap_or("").to_uppercase();
            let src_row = by_sym.get(&sym).copied().or_else(|| by_isin.get(&isin).copied());
            let mut val = src.and_then(|i| src_row.and_then(|r| r.get(i))).cloned();
            if extra == "last_price" && val.as_ref().map(missing_price).unwrap_or(true) {
                val = rclose.and_then(|i| src_row.and_then(|r| r.get(i))).cloned();
            }
            row.push(match extra {
                "name" => val.unwrap_or_else(|| json!(sym)),
                "exchange" => val.unwrap_or_else(|| json!("NSE")),
                "country" => val.unwrap_or_else(|| json!("India")),
                "industry" => val.or_else(|| sector_i.and_then(|i| row.get(i)).cloned())
                    .unwrap_or_else(|| json!("UNKNOWN")),
                _ => val.unwrap_or(Value::Null),
            });
        }
    }
}

fn fill_eod_change(rows: &mut [Vec<Value>], columns: &[String]) {
    let last_i = columns.iter().position(|c| c == "last_price");
    let prev_i = columns.iter().position(|c| c == "prev_close");
    let chg_i = columns.iter().position(|c| c == "day_chg");
    let (Some(last_i), Some(prev_i), Some(chg_i)) = (last_i, prev_i, chg_i) else { return };
    for row in rows.iter_mut() {
        if !missing_num(row.get(chg_i).unwrap_or(&Value::Null)) {
            continue;
        }
        let last = json_f64(row.get(last_i).unwrap_or(&Value::Null)).unwrap_or(0.0);
        let prev = json_f64(row.get(prev_i).unwrap_or(&Value::Null)).unwrap_or(0.0);
        if last > 0.0 && prev > 0.0 {
            row[chg_i] = num(Some((last - prev) / prev * 100.0));
        }
    }
}

fn push_live(
    row: &mut Vec<Value>,
    symbol: &str,
    snaps: &HashMap<String, QuoteSnap>,
    last_fallback: f64,
    prev_fallback: Option<f64>,
    vol_fallback: Option<f64>,
) {
    let snap = snaps.get(&symbol.to_uppercase());
    let last = snap.map(|s| s.last).filter(|p| *p > 0.0).unwrap_or(last_fallback);
    let prev = snap.and_then(|s| s.prev_close).filter(|p| *p > 0.0).or(prev_fallback);
    let chg = prev.filter(|c| *c > 0.0).map(|c| (last - c) / c * 100.0);
    if last > 0.0 && row.len() > 13 {
        row[13] = num(Some(last));
    }
    row.push(snap.and_then(|s| s.token).map(|t| json!(t)).unwrap_or(Value::Null));
    row.push(num(prev));
    row.push(num(chg));
    row.push(num(snap.and_then(|s| s.volume).or(vol_fallback)));
}

struct Position {
    qty: f64,
    price: f64,
    value: f64,
    weight: f64,
    target_w: f64,
    target_qty: f64,
    delta_qty: f64,
    delta_value: f64,
    pnl: f64,
}

fn suggestion_columns() -> Vec<String> {
    [
        "symbol", "isin", "action", "sleeve", "rank", "composite", "quality", "value",
        "momentum", "low_vol", "sector", "held", "current_qty", "last_price",
        "current_value", "current_weight", "target_weight", "target_qty",
        "delta_qty", "delta_value", "conviction", "rules", "reason", "pnl", "asset_class",
        "kite_token", "prev_close", "day_chg", "volume",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn live_from_table(holdings: &TableResponse) -> Vec<LiveName> {
    let isins = column_str(holdings, "isin");
    let symbols = column_str(holdings, "symbol");
    let qty = column_f64(holdings, "current_qty");
    let px = column_f64(holdings, "price");
    let n = holdings.n_rows;
    let mut out = Vec::new();
    for i in 0..n {
        let isin = isins.get(i).cloned().flatten().unwrap_or_default();
        let symbol = symbols.get(i).cloned().flatten().unwrap_or_default();
        if isin.is_empty() && symbol.is_empty() {
            continue;
        }
        out.push(LiveName {
            symbol: symbol.to_uppercase(),
            isin: isin.to_uppercase(),
            qty: qty.get(i).copied().flatten().unwrap_or(0.0),
            last_price: px.get(i).copied().flatten().unwrap_or(0.0),
            avg_price: 0.0,
            pnl: 0.0,
            exchange: "NSE".into(),
            asset_class: classify(&isin, &symbol, "NSE").into(),
        });
    }
    out
}

struct BookIndex {
    by_isin: HashMap<String, LiveName>,
    by_symbol: HashMap<String, LiveName>,
}

fn index_live(live: &[LiveName]) -> BookIndex {
    let mut by_isin = HashMap::new();
    let mut by_symbol = HashMap::new();
    for n in live {
        if !n.isin.is_empty() {
            by_isin.insert(n.isin.clone(), n.clone());
        }
        if !n.symbol.is_empty() {
            by_symbol.insert(n.symbol.clone(), n.clone());
        }
    }
    BookIndex { by_isin, by_symbol }
}

fn lookup<'a>(book: &'a BookIndex, isin: &str, symbol: &str) -> Option<&'a LiveName> {
    let i = isin.trim().to_uppercase();
    let s = symbol.trim().to_uppercase();
    if !i.is_empty() {
        if let Some(n) = book.by_isin.get(&i) {
            return Some(n);
        }
    }
    if !s.is_empty() {
        return book.by_symbol.get(&s);
    }
    None
}

fn key(isin: &str, symbol: &str) -> String {
    let i = isin.trim().to_uppercase();
    if !i.is_empty() {
        return format!("I:{i}");
    }
    format!("S:{}", symbol.trim().to_uppercase())
}

fn size_row(
    live: Option<&LiveName>,
    symbol: &str,
    quotes: &HashMap<String, f64>,
    nav: f64,
    target_w: f64,
) -> Position {
    let qty = live.map(|n| n.qty).unwrap_or(0.0);
    let mut price = live.map(|n| n.last_price).unwrap_or(0.0);
    if price <= 0.0 {
        if let Some(p) = quotes.get(&symbol.to_uppercase()) {
            price = *p;
        }
    }
    let value = if price > 0.0 { qty * price } else { 0.0 };
    let weight = if nav > 0.0 { value / nav } else { 0.0 };
    let target_qty = if price > 0.0 && nav > 0.0 {
        (target_w * nav / price).floor()
    } else {
        0.0
    };
    let delta_qty = target_qty - qty;
    let delta_value = delta_qty * price;
    let avg = live.map(|n| n.avg_price).unwrap_or(0.0);
    let pnl = live.map(|n| n.pnl).filter(|p| *p != 0.0).unwrap_or_else(|| {
        if avg > 0.0 && price > 0.0 { (price - avg) * qty } else { 0.0 }
    });
    Position {
        qty, price, value, weight, target_w, target_qty, delta_qty, delta_value,
        pnl,
    }
}

fn target_weights(selected: &[(String, f64)], max_w: f64) -> HashMap<String, f64> {
    let n = selected.len();
    if n == 0 {
        return HashMap::new();
    }
    let cap = if n as f64 * max_w < 1.0 { 1.0 / n as f64 } else { max_w };
    let total: f64 = selected.iter().map(|(_, s)| s).sum();
    let mut w: HashMap<String, f64> = selected
        .iter()
        .map(|(k, s)| {
            let raw = if total > 0.0 { s / total } else { 1.0 / n as f64 };
            (k.clone(), raw.min(cap))
        })
        .collect();
    let sum: f64 = w.values().sum();
    if sum > 1.0 && sum > 0.0 {
        for v in w.values_mut() {
            *v /= sum;
        }
    }
    w
}

fn apply_targets(
    rows: &mut [Vec<Value>],
    targets: &HashMap<String, f64>,
    quotes: &HashMap<String, f64>,
    nav: f64,
) {
    for row in rows.iter_mut() {
        let symbol = row[0].as_str().unwrap_or("").to_string();
        let isin = row[1].as_str().unwrap_or("").to_string();
        let action = row[2].as_str().unwrap_or("").to_string();
        let tw = match action.as_str() {
            "SELL" | "PASS" | "WATCH" => 0.0,
            _ => *targets.get(&key(&isin, &symbol)).unwrap_or(&0.0),
        };
        let qty = json_f64(&row[12]).unwrap_or(0.0);
        let mut price = json_f64(&row[13]).unwrap_or(0.0);
        if price <= 0.0 {
            price = quotes.get(&symbol.to_uppercase()).copied().unwrap_or(0.0);
        }
        if action == "HOLD" && tw == 0.0 {
            row[16] = num(Some(if nav > 0.0 { json_f64(&row[14]).unwrap_or(0.0) / nav } else { 0.0 }));
            row[17] = num(Some(qty));
            row[18] = num(Some(0.0));
            row[19] = num(Some(0.0));
            continue;
        }
        let tgt_qty = if price > 0.0 && nav > 0.0 { (tw * nav / price).floor() } else { 0.0 };
        let mut d_qty = tgt_qty - qty;
        let mut d_val = d_qty * price;
        let full_exit = action == "SELL" && qty > 0.0;
        if !full_exit && d_val.abs() < MIN_ORDER_VALUE {
            d_qty = 0.0;
            d_val = 0.0;
        }
        row[13] = num(Some(price).filter(|p| *p > 0.0));
        row[16] = num(Some(tw).filter(|w| *w > 0.0 || action == "SELL"));
        row[17] = num(Some(tgt_qty));
        row[18] = num(Some(d_qty));
        row[19] = num(Some(d_val));
        if nav > 0.0 {
            row[15] = num(Some(json_f64(&row[14]).unwrap_or(0.0) / nav));
        }
    }
}

fn row_values(
    symbol: &str, isin: &str, action: &str, sleeve: &str, rank: f64,
    composite: Option<f64>, quality: Option<f64>, value: Option<f64>,
    momentum: Option<f64>, low_vol: Option<f64>, sector: &str, held: bool,
    pos: Position, conv: f64, rules: &str, reason: &str, asset_class: &str,
) -> Vec<Value> {
    vec![
        json!(symbol),
        json!(isin),
        json!(action),
        json!(sleeve),
        if rank.is_finite() { json!(rank.round() as i64) } else { Value::Null },
        num(composite), num(quality), num(value), num(momentum), num(low_vol),
        json!(sector),
        json!(held),
        num(Some(pos.qty)),
        num(Some(pos.price).filter(|p| *p > 0.0)),
        num(Some(pos.value).filter(|v| *v > 0.0)),
        num(Some(pos.weight).filter(|w| *w > 0.0)),
        num(Some(pos.target_w).filter(|w| *w > 0.0)),
        num(Some(pos.target_qty)),
        num(Some(pos.delta_qty)),
        num(Some(pos.delta_value)),
        json!(conv),
        json!(rules),
        json!(reason),
        num(Some(pos.pnl).filter(|p| *p != 0.0)),
        json!(asset_class),
    ]
}

fn json_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|n| n as f64))
}

fn sum_delta(table: &TableResponse, action: &str) -> f64 {
    let Some(ai) = table.columns.iter().position(|c| c == "action") else { return 0.0 };
    let Some(di) = table.columns.iter().position(|c| c == "delta_value") else { return 0.0 };
    table.rows.iter().filter(|r| r.get(ai).and_then(Value::as_str) == Some(action))
        .filter_map(|r| r.get(di).and_then(json_f64))
        .sum()
}

fn decide(
    is_held: bool,
    rank: f64,
    top_n: usize,
    buffer_n: usize,
    room: usize,
    buy_used: &mut usize,
    momentum: Option<f64>,
    quality: Option<f64>,
    value: Option<f64>,
    low_vol: Option<f64>,
    abs_mom: Option<f64>,
    profitability: Option<f64>,
    mom_on: bool,
    gate_on: bool,
    quality_on: bool,
    lv_on: bool,
    dual_on: bool,
    asset: &str,
    book_loaded: bool,
) -> (Option<String>, String, String) {
    let top = top_n as f64;
    let buf = buffer_n as f64;

    if asset == "mf" {
        return (
            Some(if is_held { "HOLD".into() } else { "WATCH".into() }),
            "mf_satellite".into(),
            "Mutual-fund folio: satellite holding. Prefer a listed ETF when adding.".into(),
        );
    }
    if is_held && rank > buf {
        return (
            Some("SELL".into()),
            "rank_buffer_exit".into(),
            format!("Rank {rank:.0} is past the sell line ({buffer_n}). The buffer no longer protects this holding."),
        );
    }
    if is_held && rank > top {
        if asset == "equity" && profitability.map(|p| p < 0.0).unwrap_or(false) {
            return (
                Some("SELL".into()),
                "profitability_broke".into(),
                format!("Rank {rank:.0} left the top {top_n} and IIMA profitability turned negative. Close — the quality thesis is broken."),
            );
        }
    }
    if is_held && rank <= buf {
        let reason = if rank <= top {
            format!("Rank {rank:.0} is still inside the top {top_n}. Keep.")
        } else {
            format!("Rank {rank:.0} is outside the top {top_n} but inside the buffer (sell only after {buffer_n}). Keep.")
        };
        return (Some("HOLD".into()), "rank_buffer_keep".into(), reason);
    }
    if !is_held && rank <= top {
        if !book_loaded {
            return (
                Some("WATCH".into()),
                "no_book".into(),
                "In the entry zone, but no portfolio is loaded. Upload a CSV on the Book tab (or connect Kite) so BUY and SELL can both be sized against what you hold.".into(),
            );
        }
        if profitability.map(|p| p < 0.0).unwrap_or(false) && asset == "equity" {
            return (
                Some("WATCH".into()),
                "profitability_failed".into(),
                "IIMA profitability is negative. Do not open a new long — in India that print is the tunnelling signal.".into(),
            );
        }
        if mom_on && momentum.map(|m| m <= 0.0).unwrap_or(false) {
            return (
                Some("WATCH".into()),
                "momentum_failed".into(),
                "In the entry zone, but 12–1 / 6–1 momentum is non-positive. Wait for trend confirmation.".into(),
            );
        }
        if dual_on && abs_mom.map(|m| m <= 0.0).unwrap_or(false) {
            return (
                Some("WATCH".into()),
                "dual_momentum_failed".into(),
                "Relative rank is fine, but the 12-month absolute return is negative (dual momentum). Stay in cash on this ETF.".into(),
            );
        }
        if lv_on && low_vol.map(|v| v < -1.0).unwrap_or(false) {
            return (
                Some("WATCH".into()),
                "high_vol_block".into(),
                "Low-vol score is deeply negative — high-vol names crash with Indian momentum. Skip.".into(),
            );
        }
        if quality_on && quality.map(|q| q < 0.0).unwrap_or(false) {
            return (
                Some("WATCH".into()),
                "quality_failed".into(),
                "Quality (QMJ) is negative. Indian evidence puts quality first; do not start a new long.".into(),
            );
        }
        if gate_on {
            if let (Some(q), Some(v)) = (quality, value) {
                if v > 0.0 && q < 0.0 {
                    return (
                        Some("WATCH".into()),
                        "value_quality_gate".into(),
                        "Cheap (positive value) but poor quality — the value×quality gate blocks a new buy.".into(),
                    );
                }
            }
        }
        if *buy_used < room {
            *buy_used += 1;
            let left = room - *buy_used;
            let extra = if left > 0 {
                format!(" ({left} more after this name).")
            } else {
                ".".into()
            };
            return (
                Some("BUY".into()),
                if dual_on { "rank_buffer_entry;dual_momentum".into() } else { "rank_buffer_entry".into() },
                format!("Rank {rank:.0} entered the top {top_n} and a slot is free{extra}"),
            );
        }
        return (
            Some("WATCH".into()),
            "book_full".into(),
            format!("Rank {rank:.0} is in the entry zone but the book is full. Wait for a sell."),
        );
    }
    if !is_held && rank <= buf {
        return (
            Some("WATCH".into()),
            "buffer_zone".into(),
            format!("Rank {rank:.0} is inside the buffer band (≤ {buffer_n}) but not yet in the top {top_n}."),
        );
    }
    (None, String::new(), String::new())
}

fn conviction(action: &str, composite: Option<f64>, rank: f64, top_n: usize, buffer_n: usize) -> f64 {
    if action == "SELL" {
        let span = (buffer_n as f64).max(1.0);
        return ((rank - buffer_n as f64) / span).clamp(0.0, 1.0);
    }
    if let Some(c) = composite {
        return (0.5 + c / 4.0).clamp(0.0, 1.0);
    }
    (1.0 - (rank - 1.0) / (top_n.max(1) as f64)).max(0.0)
}

fn filter_action(table: &TableResponse, action: &str) -> TableResponse {
    let Some(idx) = table.columns.iter().position(|c| c == "action") else {
        return TableResponse::empty("no action column");
    };
    let rows: Vec<Vec<Value>> = table
        .rows
        .iter()
        .filter(|r| r.get(idx).and_then(Value::as_str) == Some(action))
        .cloned()
        .collect();
    TableResponse {
        n_rows: rows.len(),
        columns: table.columns.clone(),
        rows,
        note: None,
    }
}

fn filter_queued(table: &TableResponse, top_n: f64) -> TableResponse {
    let Some(ai) = table.columns.iter().position(|c| c == "action") else {
        return TableResponse::empty("no action column");
    };
    let Some(ri) = table.columns.iter().position(|c| c == "rank") else {
        return TableResponse::empty("no rank column");
    };
    let rows: Vec<Vec<Value>> = table
        .rows
        .iter()
        .filter(|r| {
            r.get(ai).and_then(Value::as_str) == Some("WATCH")
                && r.get(ri).and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|n| n as f64)))
                    .map(|rank| rank <= top_n)
                    .unwrap_or(false)
        })
        .cloned()
        .collect();
    TableResponse {
        n_rows: rows.len(),
        columns: table.columns.clone(),
        rows,
        note: None,
    }
}

fn action_ord(a: &str) -> u8 {
    match a {
        "SELL" => 0,
        "BUY" => 1,
        "WATCH" => 2,
        "HOLD" => 3,
        "PASS" => 8,
        _ => 9,
    }
}

fn num(v: Option<f64>) -> Value {
    match v {
        Some(x) if x.is_finite() => json!(x),
        _ => Value::Null,
    }
}

fn rule_copy(top_n: usize, buffer_n: usize, mom_on: bool, q_gate: bool) -> Vec<Map<String, Value>> {
    let mut rules = vec![
        rule("rank_buffer_entry", "BUY",
            format!("Buy names that enter the top {top_n} when a slot is free.")),
        rule("rank_buffer_exit", "SELL",
            format!("Sell a holding only after its rank falls past {buffer_n} (top {top_n} × buffer). Oscillation around {top_n} is ignored.")),
        rule("rank_buffer_keep", "HOLD",
            format!("Keep holdings ranked {}–{buffer_n}. The buffer exists to cut turnover and STCG.", top_n + 1)),
    ];
    if mom_on {
        rules.push(rule("momentum_failed", "WATCH",
            "A new buy needs positive 12–1 / 6–1 momentum.".into()));
    }
    if q_gate {
        rules.push(rule("value_quality_gate", "WATCH",
            "Do not buy cheap-and-junk names (positive value, negative quality).".into()));
    }
    rules.push(rule("profitability_broke", "SELL",
        "Close a holding that has left the top-N if IIMA profitability turns negative (quality thesis broken). Do not use a percent stop-loss.".into()));
    rules.push(rule("trend_overlay", "HOLD",
        "Scale the whole book with the 200-day MA (Moskowitz absolute momentum, ±2% hysteresis). Portfolio overlay, not a stock stop.".into()));
    rules.push(rule("tax_defer", "HOLD",
        "Defer a rank-buffer sell if the lot is inside 30 days of 12-month LTCG (STCG 20% vs LTCG 12.5%), unless rank is already past the sell line.".into()));
    rules.push(rule("dual_momentum", "BUY",
        "ETFs need dual momentum: relative rank plus a positive 12-month return.".into()));
    rules.push(rule("mf_satellite", "HOLD",
        "Mutual-fund folios stay as satellite holdings; prefer a listed ETF to add.".into()));
    rules.push(rule("high_vol_block", "WATCH",
        "Skip names with a deeply negative low-vol score (momentum crash filter).".into()));
    rules.push(rule("book_full", "WATCH",
        "Entry-zone names wait if every current holding is still inside the buffer.".into()));
    rules.push(rule("no_book", "WATCH",
        "Without a loaded book the screen does not emit BUY. Upload your portfolio or connect Kite so sells can fire on what you hold and buys can fill free slots.".into()));
    rules.push(rule("outside_universe", "SELL",
        "A holding that is not in the current research universe is flagged to review and sell if it no longer belongs.".into()));
    rules.push(rule("live_move", "WATCH",
        "Live tape: a name lights once when |last / prev_close − 1| first exceeds 3% (Databento-style latch).".into()));
    rules
}

fn rule(id: &str, action: &str, text: String) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("id".into(), json!(id));
    m.insert("action".into(), json!(action));
    m.insert("text".into(), json!(text));
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rankings(n: usize, extra: &[(&str, f64)]) -> TableResponse {
        let mut columns = vec![
            "isin".into(), "symbol".into(), "sector".into(), "rank".into(),
            "composite".into(), "momentum".into(), "quality".into(), "value".into(),
            "low_vol".into(), "last_price".into(),
        ];
        let _ = columns;
        let mut rows = Vec::new();
        for i in 0..n {
            let name = format!("N{i:02}");
            rows.push(vec![
                json!(name), json!(name), json!("Test"), json!(i + 1),
                json!(2.0 - i as f64 * 0.1), json!(1.0), json!(0.5), json!(0.2),
                json!(0.0), json!(100.0),
            ]);
        }
        for (sym, rank) in extra {
            rows.push(vec![
                json!(sym), json!(sym), json!("Test"), json!(rank),
                json!(-1.0), json!(-0.5), json!(0.0), json!(0.0),
                json!(0.0), json!(50.0),
            ]);
        }
        let n_rows = rows.len();
        TableResponse {
            columns: vec![
                "isin".into(), "symbol".into(), "sector".into(), "rank".into(),
                "composite".into(), "momentum".into(), "quality".into(), "value".into(),
                "low_vol".into(), "last_price".into(),
            ],
            rows,
            n_rows,
            note: None,
        }
    }

    fn cfg() -> Value {
        json!({
            "sleeve_a": {
                "top_n": 5,
                "rank_buffer_multiple": 2.0,
                "max_weight_per_stock": 0.25,
                "factors": {
                    "weight_momentum": 0.0,
                    "weight_quality": 0.0,
                    "weight_low_vol": 0.0,
                    "value_quality_gate": false
                }
            }
        })
    }

    fn actions(table: &TableResponse) -> Vec<(String, String)> {
        let ai = table.columns.iter().position(|c| c == "action").unwrap();
        table.rows.iter().filter_map(|r| {
            Some((
                r.first()?.as_str()?.to_string(),
                r.get(ai)?.as_str()?.to_string(),
            ))
        }).collect()
    }

    #[test]
    fn empty_book_does_not_emit_buy_only() {
        let ranks = rankings(8, &[]);
        let out = suggest(&ranks, &[], &HashMap::new(), &cfg(), 0.0, "empty", "not connected", &HashMap::new());
        assert_eq!(out.buys.n_rows, 0, "no book → no sized BUY");
        assert_eq!(out.sells.n_rows, 0, "no book → no SELL");
        let watch = actions(&out.watch);
        assert!(watch.iter().any(|(s, a)| s == "N00" && a == "WATCH"));
    }

    #[test]
    fn uploaded_holding_past_buffer_is_a_sell_and_frees_a_buy() {
        let ranks = rankings(8, &[("OLD", 12.0)]);
        let live = vec![
            LiveName {
                symbol: "N00".into(), isin: "N00".into(), qty: 10.0,
                last_price: 100.0, avg_price: 90.0, pnl: 0.0,
                exchange: "NSE".into(), asset_class: "equity".into(),
            },
            LiveName {
                symbol: "OLD".into(), isin: "OLD".into(), qty: 4.0,
                last_price: 50.0, avg_price: 80.0, pnl: 0.0,
                exchange: "NSE".into(), asset_class: "equity".into(),
            },
        ];
        let out = suggest(
            &ranks, &live, &HashMap::new(), &cfg(), 0.0,
            "upload", "uploaded portfolio", &HashMap::new(),
        );
        let sells: Vec<_> = actions(&out.sells).into_iter().map(|(s, _)| s).collect();
        let buys: Vec<_> = actions(&out.buys).into_iter().map(|(s, _)| s).collect();
        assert!(sells.contains(&"OLD".to_string()), "held name past buffer must SELL, got {sells:?}");
        assert!(!buys.is_empty(), "sell frees a slot so a BUY should appear, got {buys:?}");
        assert!(out.summary.get("n_sells").and_then(Value::as_u64).unwrap_or(0) >= 1);
        assert!(out.summary.get("n_buys").and_then(Value::as_u64).unwrap_or(0) >= 1);
    }
}

