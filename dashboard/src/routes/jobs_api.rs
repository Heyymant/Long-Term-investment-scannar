//! Endpoints that launch Python analytics runs.
//!
//! These run *local analytics only*. There is no path from here to an order.

use std::path::PathBuf;

use axum::extract::State;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::AppError;
use crate::data::models::RunRequest;
use crate::state::SharedState;

/// POST /api/backtest/run - run a strategy design over history.
pub async fn run_backtest(
    State(state): State<SharedState>,
    Json(req): Json<RunRequest>,
) -> Result<Json<Value>, AppError> {
    let mut args: Vec<String> = Vec::new();

    if let Some(preset) = &req.preset {
        args.push("--preset".into());
        args.push(preset.clone());
    } else if let Some(config) = &req.config {
        let path = write_config(&state, config)?;
        args.push("--config".into());
        args.push(path.to_string_lossy().to_string());
    } else {
        return Err(AppError::bad_request("provide either `preset` or `config`"));
    }

    args.push("--engine".into());
    args.push(req.engine.clone());
    if let Some(source) = &req.source {
        args.push("--source".into());
        args.push(source.clone());
    }
    if !req.validate {
        args.push("--no-validate".into());
    }

    let job_id = state.jobs.run_backtest(args).await;
    Ok(Json(json!({
        "job_id": job_id,
        "status": "queued",
        "stream": "/ws/jobs",
    })))
}

#[derive(Debug, Deserialize)]
pub struct OptimizeRequest {
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub config: Option<Value>,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_trials")]
    pub n: u32,
    #[serde(default)]
    pub source: Option<String>,
}

fn default_method() -> String {
    "random".into()
}

fn default_trials() -> u32 {
    30
}

/// POST /api/optimize/run
pub async fn run_optimization(
    State(state): State<SharedState>,
    Json(req): Json<OptimizeRequest>,
) -> Result<Json<Value>, AppError> {
    let mut args: Vec<String> = Vec::new();

    if let Some(preset) = &req.preset {
        args.push("--preset".into());
        args.push(preset.clone());
    } else if let Some(config) = &req.config {
        let path = write_config(&state, config)?;
        args.push("--config".into());
        args.push(path.to_string_lossy().to_string());
    } else {
        return Err(AppError::bad_request("provide either `preset` or `config`"));
    }

    args.extend([
        "--method".into(), req.method.clone(),
        "--n".into(), req.n.to_string(),
    ]);
    if let Some(source) = &req.source {
        args.push("--source".into());
        args.push(source.clone());
    }

    let job_id = state.jobs.run_optimization(args).await;
    Ok(Json(json!({ "job_id": job_id, "status": "queued", "stream": "/ws/jobs" })))
}

/// POST /api/validation/run
pub async fn run_validation(
    State(state): State<SharedState>,
    Json(req): Json<RunRequest>,
) -> Result<Json<Value>, AppError> {
    let preset = req.preset.clone().unwrap_or_else(|| "full_composite".into());
    let mut args = vec!["--preset".to_string(), preset, "--compare-presets".to_string()];
    if let Some(source) = &req.source {
        args.push("--source".into());
        args.push(source.clone());
    }
    let job_id = state.jobs.run_validation(args).await;
    Ok(Json(json!({ "job_id": job_id, "status": "queued", "stream": "/ws/jobs" })))
}

/// POST /api/rebalance/run - regenerate the manual checklist.
pub async fn run_rebalance(
    State(state): State<SharedState>,
    Json(req): Json<RunRequest>,
) -> Result<Json<Value>, AppError> {
    let preset = req.preset.clone().unwrap_or_else(|| "full_composite".into());
    let mut args = vec!["--preset".to_string(), preset];
    if let Some(source) = &req.source {
        args.push("--source".into());
        args.push(source.clone());
    }
    let job_id = state.jobs.run_rebalance(args).await;
    Ok(Json(json!({
        "job_id": job_id,
        "status": "queued",
        "note": "Generates a checklist only. No orders are placed."
    })))
}

#[derive(Debug, Deserialize, Default)]
pub struct ResearchRequest {
    #[serde(default)]
    pub full: bool,
}

/// POST /api/research/run - pytest + ranking refresh; `--full` re-runs the horse race.
pub async fn run_research(
    State(state): State<SharedState>,
    Json(req): Json<ResearchRequest>,
) -> Result<Json<Value>, AppError> {
    let args = if req.full {
        vec!["--full".to_string()]
    } else {
        vec!["--quick".to_string()]
    };
    let job_id = state.jobs.run_research(args).await;
    Ok(Json(json!({
        "job_id": job_id,
        "status": "queued",
        "mode": if req.full { "full" } else { "quick" },
        "stream": "/ws/jobs",
        "note": "Local analytics only. Does not change the dashboard latest pointer or place orders.",
    })))
}

/// GET /api/jobs
pub async fn list_jobs(State(state): State<SharedState>) -> Json<Value> {
    Json(json!({ "jobs": state.jobs.list().await }))
}

/// Persist an inline strategy design so the Python CLI can read it.
fn write_config(state: &SharedState, config: &Value) -> Result<PathBuf, AppError> {
    let dir = state.config.artifacts_dir.clone();
    std::fs::create_dir_all(&dir)
        .map_err(|e| AppError::internal(format!("cannot create artifacts dir: {e}")))?;

    let path = dir.join("strategy_config.json");
    let text = serde_json::to_string_pretty(config)
        .map_err(|e| AppError::bad_request(format!("invalid config JSON: {e}")))?;
    std::fs::write(&path, text)
        .map_err(|e| AppError::internal(format!("cannot write strategy config: {e}")))?;
    Ok(path)
}
