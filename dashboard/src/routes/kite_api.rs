//! Kite endpoints - strictly read-only, plus the daily login flow.

use axum::extract::{Query, State};
use axum::response::{Html, IntoResponse, Redirect};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use super::AppError;
use crate::data::models::{Holding, HoldingsResponse};
use crate::state::SharedState;

#[derive(Debug, Deserialize)]
pub struct CallbackQuery {
    pub request_token: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoginBody {
    pub request_token: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct QuoteQuery {
    /// Comma-separated `EXCHANGE:SYMBOL` list, e.g. `NSE:INFY,NSE:TCS`.
    pub instruments: Option<String>,
}

/// GET /kite/login - bounce the user to Zerodha's login page.
pub async fn login_redirect(State(state): State<SharedState>) -> impl IntoResponse {
    let guard = state.kite.read().await;
    match guard.as_ref() {
        Some(client) => Redirect::to(&client.login_url()).into_response(),
        None => Html(
            "<h3>Kite is not configured</h3>\
             <p>Set <code>KITE_API_KEY</code> and <code>KITE_API_SECRET</code> in \
             <code>.env</code>, then restart the dashboard.</p>",
        )
        .into_response(),
    }
}

/// GET /kite/callback - Zerodha redirects here with a request_token.
pub async fn callback(
    State(state): State<SharedState>,
    Query(q): Query<CallbackQuery>,
) -> impl IntoResponse {
    let Some(request_token) = q.request_token else {
        return Html(format!(
            "<h3>Login failed</h3><p>No request token returned (status: {}).</p>",
            q.status.unwrap_or_else(|| "unknown".into())
        ))
        .into_response();
    };

    let mut guard = state.kite.write().await;
    let Some(client) = guard.as_mut() else {
        return Html("<h3>Kite is not configured</h3>").into_response();
    };

    match client.generate_session(&request_token).await {
        Ok(_) => {
            info!("Kite session established (token valid for today)");
            Html(
                "<h3>Connected to Zerodha</h3>\
                 <p>The access token is valid for today only; you will need to log in \
                 again tomorrow.</p>\
                 <p><a href=\"/\">Back to the dashboard</a></p>",
            )
            .into_response()
        }
        Err(e) => {
            warn!("Kite login failed: {e}");
            Html(format!("<h3>Login failed</h3><pre>{e}</pre>")).into_response()
        }
    }
}

/// POST /api/kite/login - programmatic token exchange.
pub async fn login_post(
    State(state): State<SharedState>,
    Json(body): Json<LoginBody>,
) -> Result<Json<Value>, AppError> {
    let mut guard = state.kite.write().await;
    let client = guard
        .as_mut()
        .ok_or_else(|| AppError::bad_request("Kite is not configured"))?;

    client
        .generate_session(&body.request_token)
        .await
        .map_err(|e| AppError::bad_request(format!("login failed: {e}")))?;

    Ok(Json(json!({ "status": "ok", "message": "authenticated for today" })))
}

/// GET /api/holdings - live holdings joined with the latest target weights.
pub async fn holdings(State(state): State<SharedState>) -> Result<Json<HoldingsResponse>, AppError> {
    let targets = load_targets(&state);

    let client = match state.kite_client().await {
        Ok(c) => c,
        Err(msg) => {
            // Not an error state: show targets and explain what is missing.
            return Ok(Json(HoldingsResponse {
                holdings: vec![],
                total_value: 0.0,
                total_pnl: 0.0,
                targets,
                note: Some(msg),
            }));
        }
    };

    let raw = client
        .holdings()
        .await
        .map_err(|e| AppError::unavailable(format!("could not fetch holdings: {e}"), "Check your Kite session"))?;

    let holdings: Vec<Holding> = serde_json::from_value(raw).unwrap_or_default();
    let total_value: f64 = holdings.iter().map(|h| h.quantity * h.last_price).sum();
    let total_pnl: f64 = holdings.iter().map(|h| h.pnl).sum();

    Ok(Json(HoldingsResponse {
        holdings,
        total_value,
        total_pnl,
        targets,
        note: None,
    }))
}

/// GET /api/positions
pub async fn positions(State(state): State<SharedState>) -> Result<Json<Value>, AppError> {
    let client = state
        .kite_client()
        .await
        .map_err(|m| AppError::unavailable(m, "Visit /kite/login"))?;
    let data = client
        .positions()
        .await
        .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(Json(data))
}

/// GET /api/quotes?instruments=NSE:INFY,NSE:TCS
pub async fn quotes(
    State(state): State<SharedState>,
    Query(q): Query<QuoteQuery>,
) -> Result<Json<Value>, AppError> {
    let client = state
        .kite_client()
        .await
        .map_err(|m| AppError::unavailable(m, "Visit /kite/login"))?;

    let instruments: Vec<String> = q
        .instruments
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if instruments.is_empty() {
        return Err(AppError::bad_request(
            "pass ?instruments=NSE:INFY,NSE:TCS",
        ));
    }

    let data = client
        .ltp(&instruments)
        .await
        .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(Json(data))
}

/// GET /api/trades - executed trades for the journal (read-only).
pub async fn trades(State(state): State<SharedState>) -> Result<Json<Value>, AppError> {
    let client = state
        .kite_client()
        .await
        .map_err(|m| AppError::unavailable(m, "Visit /kite/login"))?;
    let data = client
        .trades()
        .await
        .map_err(|e| AppError::internal(e.to_string()))?;
    Ok(Json(data))
}

/// Target weights from the latest run, keyed by symbol so the UI can show
/// live holdings against their targets.
fn load_targets(state: &SharedState) -> std::collections::HashMap<String, f64> {
    use crate::data::artifacts::{column_f64, column_str};

    let mut out = std::collections::HashMap::new();
    let Ok(table) = state.artifacts.read_table(None, "holdings_target.csv") else {
        return out;
    };

    let symbols = column_str(&table, "symbol");
    let weights = column_f64(&table, "weight");

    for (symbol, weight) in symbols.into_iter().zip(weights) {
        if let (Some(s), Some(w)) = (symbol, weight) {
            out.insert(s, w);
        }
    }
    out
}
