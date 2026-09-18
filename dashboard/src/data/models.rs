//! Serde models shared with the frontend.
//!
//! These mirror the artifacts contract emitted by the Python side
//! (`src/reports/artifacts.py`). `schema_version` is carried through so a
//! mismatch surfaces loudly instead of silently misreading columns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const SUPPORTED_SCHEMA: &str = "1.0.0";

/// Reserved for future run-metadata responses.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunInfo {
    pub run_id: String,
    #[serde(default)]
    pub schema_version: String,
}

/// A time series formatted for uPlot: parallel arrays, not row objects.
/// uPlot wants columnar data and this avoids a per-point allocation in JS.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SeriesData {
    /// Unix timestamps in seconds.
    pub t: Vec<i64>,
    pub series: HashMap<String, Vec<Option<f64>>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct EquityCurveResponse {
    pub run_id: String,
    pub data: SeriesData,
    pub n_points: usize,
}

/// Typed view of backtest.json (routes currently pass it through as raw JSON).
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsResponse {
    #[serde(default)]
    pub schema_version: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub sleeves: HashMap<String, serde_json::Value>,
}

/// Generic tabular payload for the frontend's table component.
#[derive(Debug, Clone, Serialize, Default)]
pub struct TableResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub n_rows: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl TableResponse {
    pub fn empty(note: &str) -> Self {
        Self {
            columns: vec![],
            rows: vec![],
            n_rows: 0,
            note: Some(note.to_string()),
        }
    }
}

// --------------------------------------------------------------------------
// Kite (read-only)
// --------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Holding {
    pub tradingsymbol: String,
    #[serde(default)]
    pub isin: String,
    #[serde(default)]
    pub quantity: f64,
    #[serde(default)]
    pub average_price: f64,
    #[serde(default)]
    pub last_price: f64,
    #[serde(default)]
    pub pnl: f64,
    #[serde(default)]
    pub day_change_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct HoldingsResponse {
    pub holdings: Vec<Holding>,
    pub total_value: f64,
    pub total_pnl: f64,
    /// Target weights joined in from the latest run, so the UI can show drift.
    pub targets: HashMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// One market tick relayed from the Kite WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub instrument_token: u32,
    pub last_price: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ohlc_close: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change: Option<f64>,
}

// --------------------------------------------------------------------------
// Jobs
// --------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobUpdate {
    pub job_id: String,
    pub status: JobStatus,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RunRequest {
    /// Full strategy design, validated against the shared JSON Schema.
    #[serde(default)]
    pub config: Option<serde_json::Value>,
    /// Or a named preset from the Python side.
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default = "default_engine")]
    pub engine: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub validate: bool,
}

fn default_engine() -> String {
    "vectorized".to_string()
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl ApiError {
    pub fn new(error: impl Into<String>) -> Self {
        Self { error: error.into(), hint: None }
    }

    pub fn with_hint(error: impl Into<String>, hint: impl Into<String>) -> Self {
        Self { error: error.into(), hint: Some(hint.into()) }
    }
}
