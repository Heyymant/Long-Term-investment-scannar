//! Read-only analytics endpoints backed by the Python artifacts.

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{missing_artifact, AppError};
use crate::book::{
    enrich_from_nse, merge_unique, normalize_symbol, parse_csv, position_rows, positions_from_kite,
    to_holdings, BookRowIn, UploadedBook,
};
use crate::data::artifacts::table_to_series;
use crate::data::models::{EquityCurveResponse, TableResponse};
use crate::signals::LiveName;
use crate::state::SharedState;

/// Cap how many rows are shipped to the browser. Tables are already small,
/// but rankings and signal logs can be long enough to hurt render time.
fn limit_table(mut table: TableResponse, limit: Option<usize>) -> TableResponse {
    if let Some(n) = limit {
        if table.rows.len() > n {
            table.rows.truncate(n);
        }
    }
    table.n_rows = table.rows.len();
    table
}

#[derive(Debug, Deserialize, Default)]
pub struct RunQuery {
    pub run_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

impl RunQuery {
    fn id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }
}

/// GET /api/backtest - headline metrics plus the equity curve.
pub async fn backtest(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let metrics = state
        .artifacts
        .read_json(q.id(), "backtest.json")
        .map_err(|e| missing_artifact("backtest metrics", &e))?;

    let curve = state
        .artifacts
        .read_table(q.id(), "equity_curve.parquet").ok()
        .and_then(|t| table_to_series(&t, "date"));

    let run_id = q
        .run_id
        .clone()
        .or_else(|| state.artifacts.latest_run_id())
        .unwrap_or_default();

    Ok(Json(json!({
        "run_id": run_id,
        "metrics": metrics,
        "equity_curve": curve.map(|data| EquityCurveResponse {
            n_points: data.t.len(),
            run_id: run_id.clone(),
            data,
        }),
    })))
}

/// GET /api/attribution - true alpha and factor betas.
pub async fn attribution(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let attribution = state
        .artifacts
        .read_json(q.id(), "attribution.json")
        .map_err(|e| missing_artifact("attribution", &e))?;

    let ic = state
        .artifacts
        .read_table(q.id(), "ic_icir.parquet")
        .ok()
        .map(|t| limit_table(t, Some(50)));

    Ok(Json(json!({ "attribution": attribution, "ic_icir": ic })))
}

/// GET /api/validation - the statistical verdict.
pub async fn validation(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let stats = state
        .artifacts
        .read_json_any(q.id(), "validation_stats.json")
        .map_err(|e| missing_artifact("validation stats", &e))?;
    Ok(Json(stats))
}

/// GET /api/optimize - parameter sweep results.
pub async fn optimize(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let table = state
        .artifacts
        .read_table(q.id(), "optimization_results.parquet")
        .map(|t| limit_table(t, q.limit.or(Some(500))))
        .unwrap_or_else(|_| TableResponse::empty("no optimization run yet"));

    let summary = state
        .artifacts
        .read_json_any(q.id(), "optimization_summary.json")
        .unwrap_or(Value::Null);

    Ok(Json(json!({ "results": table, "summary": summary })))
}

/// GET /api/rankings - current factor scores.
pub async fn rankings(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<TableResponse>, AppError> {
    let df = state
        .artifacts
        .read_table(q.id(), "rankings.parquet")
        .map_err(|e| missing_artifact("rankings", &e))?;
    Ok(Json(limit_table(df, q.limit.or(Some(500)))))
}

/// GET /api/signals - screener + position suggestions vs the live Kite book.
pub async fn signals(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let mut rankings = state
        .artifacts
        .read_table(q.id(), "rankings.parquet")
        .map_err(|e| missing_artifact("rankings (needed to screen)", &e))?;
    if let Ok(etf) = state.artifacts.read_table(q.id(), "etf_rankings.parquet") {
        rankings = concat_tables(rankings, etf);
    }
    let config = state
        .artifacts
        .read_json(q.id(), "strategy_config.json")
        .unwrap_or(json!({}));

    let (mut live, cash, kite_status, book_source) = load_live_book(&state).await;
    enrich_from_nse(&mut live, &rankings);
    // Market prints come from NSE EOD on the rankings file, not from Kite.
    let mut quotes = std::collections::HashMap::new();
    let snaps: std::collections::HashMap<String, crate::signals::QuoteSnap> =
        std::collections::HashMap::new();
    for n in &live {
        if n.last_price > 0.0 {
            quotes.insert(n.symbol.clone(), n.last_price);
        }
    }

    Ok(Json(crate::signals::suggest(
        &rankings, &live, &quotes, &config, cash, &book_source, &kite_status, &snaps,
    ).into_json()))
}

async fn load_live_book(state: &SharedState) -> (Vec<LiveName>, f64, String, String) {
    {
        let uploaded = state.uploaded_book.read().await;
        if uploaded.present {
            let live = uploaded.names();
            let cash = uploaded.cash.max(0.0);
            let n = live.len();
            return (
                live,
                cash,
                format!("uploaded portfolio ({n} names)"),
                "upload".into(),
            );
        }
    }

    match state.kite_client().await {
        Err(msg) => (Vec::new(), 0.0, msg, "empty".into()),
        Ok(client) => {
            let raw = match client.holdings().await {
                Ok(v) => v,
                Err(e) => {
                    return (Vec::new(), 0.0, format!("could not fetch holdings: {e}"), "empty".into());
                }
            };
            let holdings: Vec<crate::data::models::Holding> =
                serde_json::from_value(raw).unwrap_or_default();
            let mut live: Vec<LiveName> = holdings
                .iter()
                .filter(|h| h.quantity > 0.0)
                .map(LiveName::from_holding)
                .collect();
            if let Ok(pos_raw) = client.positions().await {
                merge_unique(&mut live, positions_from_kite(&pos_raw));
            }
            if let Ok(mf_raw) = client.mf_holdings().await {
                if let Some(arr) = mf_raw.as_array() {
                    for v in arr {
                        if let Some(n) = LiveName::from_mf_json(v) {
                            live.push(n);
                        }
                    }
                }
            }
            let cash = match client.margins().await {
                Ok(m) => parse_cash(&m),
                Err(_) => 0.0,
            };
            let n = live.len();
            (live, cash, format!("authenticated ({n} holdings)"), "kite".into())
        }
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct BookUpload {
    pub cash: Option<f64>,
    pub csv: Option<String>,
    pub rows: Option<Vec<BookRowIn>>,
}

/// GET /api/book — current book (upload or Kite) with NSE EOD prints on each line.
pub async fn get_book(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let rankings = state
        .artifacts
        .read_table(q.id(), "rankings.parquet")
        .unwrap_or_else(|_| TableResponse::empty("no rankings"));
    let (mut live, cash, status, source) = load_live_book(&state).await;
    enrich_from_nse(&mut live, &rankings);
    let positions = position_rows(&live, &rankings);
    let equity: f64 = live.iter().map(LiveName::value).sum();
    Ok(Json(json!({
        "source": source,
        "status": status,
        "cash": cash,
        "equity_value": equity,
        "nav": equity + cash.max(0.0),
        "holdings": to_holdings(&live),
        "positions": positions,
        "note": if source == "empty" {
            Some("No portfolio loaded. Upload a CSV on the Book tab, or connect Kite.")
        } else {
            None
        },
    })))
}

/// POST /api/book — save an uploaded portfolio. Prices are filled from NSE EOD.
pub async fn put_book(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
    Json(body): Json<BookUpload>,
) -> Result<Json<Value>, AppError> {
    let mut rows: Vec<BookRowIn> = body.rows.unwrap_or_default();
    let mut cash = body.cash.unwrap_or(0.0).max(0.0);
    if let Some(csv) = body.csv.as_deref().filter(|s| !s.trim().is_empty()) {
        let trimmed = csv.trim_start_matches('\u{feff}').trim();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            if let Ok(parsed) = serde_json::from_str::<Vec<BookRowIn>>(trimmed) {
                rows.extend(parsed);
            } else if let Ok(wrap) = serde_json::from_str::<BookUpload>(trimmed) {
                if let Some(r) = wrap.rows {
                    rows.extend(r);
                }
                cash += wrap.cash.unwrap_or(0.0).max(0.0);
            } else {
                return Err(AppError::bad_request("could not parse JSON portfolio"));
            }
        } else {
            let (parsed, csv_cash) = parse_csv(csv).map_err(AppError::bad_request)?;
            rows.extend(parsed);
            cash += csv_cash;
        }
    }
    if rows.is_empty() && cash <= 0.0 {
        return Err(AppError::bad_request(
            "need at least one holding (symbol, qty) or a cash amount",
        ));
    }
    // Last row for a symbol wins.
    let mut by_sym: std::collections::HashMap<String, BookRowIn> = std::collections::HashMap::new();
    for mut r in rows {
        r.symbol = normalize_symbol(&r.symbol);
        r.isin = r.isin.trim().to_uppercase();
        if r.exchange.trim().is_empty() {
            r.exchange = "NSE".into();
        }
        let key = if !r.symbol.is_empty() { r.symbol.clone() } else { r.isin.clone() };
        if key.is_empty() {
            continue;
        }
        by_sym.insert(key, r);
    }
    let rows: Vec<BookRowIn> = by_sym.into_values().collect();
    let book = UploadedBook {
        present: true,
        cash,
        uploaded_at: chrono::Local::now().to_rfc3339(),
        rows,
    };
    book.save(state.artifacts.root())
        .map_err(|e| AppError::internal(format!("could not save book: {e}")))?;
    {
        let mut guard = state.uploaded_book.write().await;
        *guard = book;
    }
    get_book(State(state), Query(q)).await
}

/// DELETE /api/book — drop the uploaded portfolio (Kite holdings remain as fallback).
pub async fn clear_book(State(state): State<SharedState>) -> Json<Value> {
    UploadedBook::clear(state.artifacts.root());
    let mut guard = state.uploaded_book.write().await;
    *guard = UploadedBook::default();
    Json(json!({ "ok": true, "source": "empty" }))
}

fn parse_cash(margins: &Value) -> f64 {
    let paths = [
        "/equity/available/cash",
        "/equity/available/live_balance",
        "/equity/net",
        "/available/cash",
    ];
    for p in paths {
        if let Some(v) = margins.pointer(p).and_then(Value::as_f64) {
            return v.max(0.0);
        }
        if let Some(v) = margins.pointer(p).and_then(Value::as_i64) {
            return (v as f64).max(0.0);
        }
    }
    0.0
}

#[allow(dead_code)]
fn quote_instruments(rankings: &TableResponse, config: &Value) -> Vec<String> {
    use crate::data::artifacts::column_str;
    let top_n = config.pointer("/sleeve_a/top_n").and_then(Value::as_u64).unwrap_or(40) as usize;
    let buffer = ((top_n as f64)
        * config.pointer("/sleeve_a/rank_buffer_multiple").and_then(Value::as_f64).unwrap_or(2.0))
        .round() as usize;
    let symbols = column_str(rankings, "symbol");
    let ranks = crate::data::artifacts::column_f64(rankings, "rank");
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (i, sym) in symbols.into_iter().enumerate() {
        let Some(s) = sym else { continue };
        let rank = ranks.get(i).copied().flatten().unwrap_or(f64::MAX);
        if rank > (buffer.max(80) as f64) && out.len() >= 160 {
            continue;
        }
        let key = s.to_uppercase();
        if !seen.insert(key.clone()) {
            continue;
        }
        out.push(format!("NSE:{s}"));
        if out.len() >= 240 {
            break;
        }
    }
    out
}

#[allow(dead_code)]
fn merge_quote(
    quotes: &mut std::collections::HashMap<String, f64>,
    snaps: &mut std::collections::HashMap<String, crate::signals::QuoteSnap>,
    raw: &Value,
) {
    let Some(obj) = raw.as_object() else { return };
    for (k, v) in obj {
        let sym = k.split(':').next_back().unwrap_or(k).to_uppercase();
        let last = v.get("last_price").and_then(Value::as_f64).unwrap_or(0.0);
        let token = v.get("instrument_token").and_then(Value::as_u64).map(|n| n as u32);
        let prev = v.pointer("/ohlc/close").and_then(Value::as_f64).filter(|c| *c > 0.0);
        let volume = v.get("volume").and_then(Value::as_f64)
            .or_else(|| v.get("volume").and_then(Value::as_u64).map(|n| n as f64));
        if last > 0.0 {
            quotes.insert(sym.clone(), last);
        }
        snaps.insert(sym, crate::signals::QuoteSnap { last, prev_close: prev, token, volume });
    }
}

#[allow(dead_code)]
async fn publish_directory(
    state: &crate::state::SharedState,
    snaps: &std::collections::HashMap<String, crate::signals::QuoteSnap>,
) {
    let mut tokens = Vec::new();
    {
        let mut dir = state.symbol_directory.write().await;
        for (sym, snap) in snaps {
            if let Some(t) = snap.token {
                dir.insert(t, sym.clone());
                tokens.push(t);
            }
        }
    }
    if !tokens.is_empty() {
        let mut watch = state.watch_tokens.write().await;
        for t in tokens {
            if !watch.contains(&t) {
                watch.push(t);
            }
        }
    }
}

#[allow(dead_code)]
fn merge_ltp(quotes: &mut std::collections::HashMap<String, f64>, raw: &Value) {
    let Some(obj) = raw.as_object() else { return };
    for (k, v) in obj {
        let px = v.get("last_price").and_then(Value::as_f64)
            .or_else(|| v.as_f64());
        let Some(px) = px.filter(|p| *p > 0.0) else { continue };
        let sym = k.split(':').next_back().unwrap_or(k).to_uppercase();
        quotes.entry(sym).or_insert(px);
    }
}

fn concat_tables(a: TableResponse, b: TableResponse) -> TableResponse {
    if b.n_rows == 0 {
        return a;
    }
    if a.n_rows == 0 {
        return b;
    }
    let mut cols = a.columns.clone();
    for c in &b.columns {
        if !cols.iter().any(|x| x == c) {
            cols.push(c.clone());
        }
    }
    let map_row = |row: &[Value], src_cols: &[String]| -> Vec<Value> {
        cols.iter().map(|c| {
            src_cols.iter().position(|x| x == c)
                .and_then(|i| row.get(i).cloned())
                .unwrap_or(Value::Null)
        }).collect()
    };
    let mut rows: Vec<Vec<Value>> = a.rows.iter().map(|r| map_row(r, &a.columns)).collect();
    rows.extend(b.rows.iter().map(|r| map_row(r, &b.columns)));
    TableResponse { n_rows: rows.len(), columns: cols, rows, note: None }
}

/// GET /api/statarb - Sleeve B pairs and signals.
pub async fn statarb(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let pairs = state
        .artifacts
        .read_table(q.id(), "statarb_pairs.parquet")
        .map(|t| limit_table(t, Some(200)))
        .unwrap_or_else(|_| TableResponse::empty("no cointegrated pairs (residual mode or none found)"));

    let signals = state
        .artifacts
        .read_table(q.id(), "statarb_signals.parquet")
        .map(|t| limit_table(t, Some(500)))
        .unwrap_or_else(|_| TableResponse::empty("Sleeve B not enabled in this run"));

    Ok(Json(json!({ "pairs": pairs, "signals": signals })))
}

/// GET /api/events - Sleeve C earnings calendar and PEAD signals.
pub async fn events(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let calendar = state
        .artifacts
        .read_table(q.id(), "earnings_calendar.parquet")
        .map(|t| limit_table(t, Some(300)))
        .unwrap_or_else(|_| TableResponse::empty("no earnings calendar loaded"));

    let signals = state
        .artifacts
        .read_table(q.id(), "event_signals.parquet")
        .map(|t| limit_table(t, Some(300)))
        .unwrap_or_else(|_| TableResponse::empty("Sleeve C not enabled in this run"));

    Ok(Json(json!({ "calendar": calendar, "signals": signals })))
}

/// GET /api/risk - exposure, drawdown, sector weights, capacity.
pub async fn risk(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let risk = state
        .artifacts
        .read_json(q.id(), "risk.json")
        .map_err(|e| missing_artifact("risk report", &e))?;

    let costs = state
        .artifacts
        .read_table(q.id(), "cost_sensitivity.parquet")
        .ok()
        .map(|t| limit_table(t, None));

    let regime = state
        .artifacts
        .read_table(q.id(), "regime_breakdown.parquet")
        .ok()
        .map(|t| limit_table(t, None));

    Ok(Json(json!({ "risk": risk, "cost_sensitivity": costs, "regime": regime })))
}

/// GET /api/rebalance - the manual execution checklist.
pub async fn rebalance(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let orders = state
        .artifacts
        .read_table(q.id(), "rebalance_orders.csv")
        .map(|t| limit_table(t, None))
        .unwrap_or_else(|_| {
            TableResponse::empty("no rebalance generated yet - run generate_rebalance.py")
        });

    let targets = state
        .artifacts
        .read_table(q.id(), "holdings_target.csv")
        .map(|t| limit_table(t, None))
        .unwrap_or_else(|_| TableResponse::empty("no target weights in this run"));

    let summary = state
        .artifacts
        .read_json_any(q.id(), "rebalance_summary.json")
        .unwrap_or(Value::Null);

    Ok(Json(json!({
        "orders": orders,
        "targets": targets,
        "summary": summary,
        "note": "Decision support only - no orders are placed. Execute manually."
    })))
}

/// GET /api/journal - reconciled fills and live-vs-backtest drift.
pub async fn journal(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let entries = state
        .artifacts
        .read_table(q.id(), "journal.parquet")
        .map(|t| limit_table(t, Some(500)))
        .unwrap_or_else(|_| TableResponse::empty("no journal yet - import your tradebook"));

    let drift = state
        .artifacts
        .read_json_any(q.id(), "live_tracking.json")
        .unwrap_or(Value::Null);

    Ok(Json(json!({ "entries": entries, "drift": drift })))
}

/// GET /api/health - data quality plus dashboard status.
pub async fn health(State(state): State<SharedState>) -> Json<Value> {
    let data_health = state
        .artifacts
        .read_json_any(None, "data_health.json")
        .unwrap_or(Value::Null);

    let kite_status = match state.kite.read().await.as_ref() {
        None => "not configured",
        Some(c) if c.has_token() => "authenticated",
        Some(_) => "needs login",
    };

    // Surface data errors outward as well as on screen.
    if let Some(errors) = data_health.get("n_errors").and_then(Value::as_u64) {
        if errors > 0 && state.alerter.enabled() {
            state
                .alerter
                .send(crate::alerts::Alert::new(
                    crate::alerts::AlertKind::DataHealth,
                    "Data quality errors detected",
                    format!("{errors} error(s) in the latest data health report"),
                ))
                .await;
        }
    }

    let (book_present, book_n) = {
        let b = state.uploaded_book.read().await;
        (b.present, b.rows.len())
    };

    Json(json!({
        "status": "ok",
        "artifacts_dir": state.artifacts.root().display().to_string(),
        "latest_run": state.artifacts.latest_run_id(),
        "kite": kite_status,
        "book": { "present": book_present, "n": book_n },
        "alerts_enabled": state.alerter.enabled(),
        "data_health": data_health,
    }))
}

/// GET /api/runs - the run registry (for reproducibility and trial counting).
pub async fn runs(State(state): State<SharedState>) -> Json<Value> {
    let ids = state.artifacts.list_runs();
    let registry = state
        .artifacts
        .read_table(None, "runs_registry.parquet")
        .map(|t| limit_table(t, Some(200)))
        .unwrap_or_else(|_| TableResponse::empty("no runs recorded yet"));

    Json(json!({
        "latest": state.artifacts.latest_run_id(),
        "run_ids": ids,
        "registry": registry,
    }))
}

/// GET /api/monthly - the year x month return grid.
pub async fn monthly(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<TableResponse>, AppError> {
    let df = state
        .artifacts
        .read_table(q.id(), "monthly_returns.parquet")
        .map_err(|e| missing_artifact("monthly returns", &e))?;
    Ok(Json(limit_table(df, None)))
}

/// GET /api/config - the strategy design behind a run.
pub async fn strategy_config(
    State(state): State<SharedState>,
    Query(q): Query<RunQuery>,
) -> Result<Json<Value>, AppError> {
    let cfg = state
        .artifacts
        .read_json(q.id(), "strategy_config.json")
        .map_err(|e| missing_artifact("strategy config", &e))?;
    Ok(Json(cfg))
}

#[derive(Debug, Deserialize, Default)]
pub struct HistoryQuery {
    pub symbol: Option<String>,
    pub run_id: Option<String>,
}

/// GET /api/history?symbol=INFY — NSE EOD last ~1 year. Never uses Kite.
pub async fn history(
    State(state): State<SharedState>,
    Query(q): Query<HistoryQuery>,
) -> Result<Json<Value>, AppError> {
    let symbol = q.symbol.unwrap_or_default().trim().to_uppercase();
    if symbol.is_empty() {
        return Err(AppError::bad_request("pass ?symbol=INFY"));
    }
    if let Some(payload) = nse_spark(&state, q.run_id.as_deref(), &symbol) {
        return Ok(Json(payload));
    }
    Err(AppError::not_found(format!(
        "No NSE EOD series for {symbol}. Run nightly research to rebuild sparks.json."
    )))
}

fn nse_spark(state: &SharedState, run_id: Option<&str>, symbol: &str) -> Option<Value> {
    let mut docs = Vec::new();
    if let Ok(v) = state.artifacts.read_json(run_id, "sparks.json") {
        docs.push(v);
    }
    let asset = state.config.assets_dir.join("sparks.json");
    if let Ok(text) = std::fs::read_to_string(asset) {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            docs.push(v);
        }
    }
    for doc in docs {
        let Some(closes) = doc.get("closes").and_then(|c| c.get(symbol)).and_then(Value::as_array) else {
            continue;
        };
        let px: Vec<Value> = closes.iter().cloned().collect();
        if px.len() < 2 {
            continue;
        }
        let t: Vec<i64> = (0..px.len() as i64).collect();
        let note = doc
            .get("note")
            .and_then(Value::as_str)
            .unwrap_or("NSE EOD last ~1 year. Decision support only.");
        return Some(json!({
            "symbol": symbol,
            "source": "nse_eod",
            "data": { "t": t, "series": { "close": px } },
            "n_points": px.len(),
            "note": note,
        }));
    }
    None
}
