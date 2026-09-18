//! Read-only analytics endpoints backed by the Python artifacts.

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::{missing_artifact, AppError};
use crate::data::artifacts::table_to_series;
use crate::data::models::{EquityCurveResponse, TableResponse};
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

    Json(json!({
        "status": "ok",
        "artifacts_dir": state.artifacts.root().display().to_string(),
        "latest_run": state.artifacts.latest_run_id(),
        "kite": kite_status,
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
