//! HTTP + WebSocket routes.

pub mod analytics;
pub mod jobs_api;
pub mod kite_api;
pub mod ws;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::data::models::ApiError;

/// Uniform error response so the frontend can always show something useful.
pub struct AppError(pub StatusCode, pub ApiError);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(self.1)).into_response()
    }
}

impl AppError {
    #[allow(dead_code)]
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self(StatusCode::NOT_FOUND, ApiError::new(msg))
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, ApiError::new(msg))
    }

    pub fn unavailable(msg: impl Into<String>, hint: impl Into<String>) -> Self {
        Self(StatusCode::SERVICE_UNAVAILABLE, ApiError::with_hint(msg, hint))
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, ApiError::new(msg))
    }
}

/// Missing artifacts are an expected state (nothing has been run yet), so we
/// return 404 with guidance rather than a 500.
pub fn missing_artifact(name: &str, err: &anyhow::Error) -> AppError {
    AppError(
        StatusCode::NOT_FOUND,
        ApiError::with_hint(
            format!("{name} unavailable: {err}"),
            "Run a backtest first: python scripts/run_backtest.py --preset full_composite",
        ),
    )
}
