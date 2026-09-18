//! Indian Equity Alpha Framework - dashboard server.
//!
//! A read-only cockpit: it reads the Python analytics artifacts, streams live
//! quotes from Zerodha Kite, and can launch local backtests. It cannot and
//! does not place orders.

mod alerts;
mod config;
mod data;
mod jobs;
mod kite;
mod routes;
mod state;

use std::time::Duration;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::config::Config;
use crate::data::artifacts::spawn_watcher;
use crate::state::{AppState, SharedState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| "alpha_dashboard=info,tower_http=warn".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = Config::from_env()?;
    print_banner(&config);

    let state = AppState::new(config.clone());
    spawn_watcher(state.artifacts.clone())?;
    spawn_ticker(state.clone());

    let app = build_router(state.clone());
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    info!("dashboard listening on http://{}", config.bind);

    axum::serve(listener, app).await?;
    Ok(())
}

fn build_router(state: SharedState) -> Router {
    use routes::{analytics, jobs_api, kite_api, ws};

    let api = Router::new()
        // Analytics (read artifacts)
        .route("/backtest", get(analytics::backtest))
        .route("/attribution", get(analytics::attribution))
        .route("/validation", get(analytics::validation))
        .route("/optimize", get(analytics::optimize))
        .route("/rankings", get(analytics::rankings))
        .route("/statarb", get(analytics::statarb))
        .route("/events", get(analytics::events))
        .route("/risk", get(analytics::risk))
        .route("/rebalance", get(analytics::rebalance))
        .route("/journal", get(analytics::journal))
        .route("/monthly", get(analytics::monthly))
        .route("/config", get(analytics::strategy_config))
        .route("/runs", get(analytics::runs))
        .route("/health", get(analytics::health))
        // Kite (read-only)
        .route("/holdings", get(kite_api::holdings))
        .route("/positions", get(kite_api::positions))
        .route("/quotes", get(kite_api::quotes))
        .route("/trades", get(kite_api::trades))
        .route("/kite/login", post(kite_api::login_post))
        // Job launchers (local analytics only)
        .route("/backtest/run", post(jobs_api::run_backtest))
        .route("/optimize/run", post(jobs_api::run_optimization))
        .route("/validation/run", post(jobs_api::run_validation))
        .route("/rebalance/run", post(jobs_api::run_rebalance))
        .route("/jobs", get(jobs_api::list_jobs))
        .layer(middleware::from_fn_with_state(state.clone(), auth_guard));

    let assets = state.config.assets_dir.clone();
    let index = assets.join("index.html");

    Router::new()
        .nest("/api", api)
        .route("/ws/ticks", get(ws::ticks))
        .route("/ws/jobs", get(ws::jobs))
        .route("/kite/login", get(kite_api::login_redirect))
        .route("/kite/callback", get(kite_api::callback))
        .fallback_service(
            ServeDir::new(&assets).not_found_service(ServeFile::new(&index)),
        )
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Bearer-token guard for /api routes.
///
/// Only enforced when DASHBOARD_AUTH_TOKEN is set. It must be set if the
/// server is bound to anything other than loopback, because this process
/// holds live broker credentials.
async fn auth_guard(
    State(state): State<SharedState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.config.auth_token.as_ref() else {
        return next.run(request).await;
    };

    let provided = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(token) if token == expected => next.run(request).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({ "error": "missing or invalid bearer token" })),
        )
            .into_response(),
    }
}

/// Start the Kite ticker if we have a session and a watchlist.
fn spawn_ticker(state: SharedState) {
    tokio::spawn(async move {
        // Give the artifact store a moment to settle after startup.
        tokio::time::sleep(Duration::from_secs(2)).await;

        let ws_url = {
            let guard = state.kite.read().await;
            match guard.as_ref().and_then(|c| c.ws_url()) {
                Some(url) => url,
                None => {
                    info!("Kite ticker not started (no session). Visit /kite/login to enable live quotes.");
                    return;
                }
            }
        };

        let tokens = watchlist_tokens(&state);
        if tokens.is_empty() {
            info!("Kite ticker not started: no instruments in the current targets");
            return;
        }

        let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
        if let Err(e) = kite::ticker::run_ticker(ws_url, tokens, state.ticks.clone(), shutdown).await {
            warn!("ticker stopped: {e}");
        }
    });
}

/// Instrument tokens to stream, taken from the latest target portfolio.
fn watchlist_tokens(state: &SharedState) -> Vec<u32> {
    use crate::data::artifacts::column_f64;

    let Ok(table) = state.artifacts.read_table(None, "holdings_target.csv") else {
        return vec![];
    };
    column_f64(&table, "kite_token")
        .into_iter()
        .flatten()
        .map(|v| v as u32)
        .collect()
}

fn print_banner(config: &Config) {
    println!();
    println!("  Indian Equity Alpha Framework - Dashboard");
    println!("  -----------------------------------------");
    println!("  bind        http://{}", config.bind);
    println!("  artifacts   {}", config.artifacts_dir.display());
    println!("  python      {} ({})", config.python_bin, config.python_dir.display());
    println!(
        "  kite        {}",
        if config.kite_configured() { "configured" } else { "not configured" }
    );
    println!(
        "  auth        {}",
        if config.auth_token.is_some() { "bearer token required" } else { "none (localhost only)" }
    );
    println!();
    println!("  READ-ONLY: this dashboard never places orders.");

    if config.is_exposed() && config.auth_token.is_none() {
        println!();
        println!("  WARNING: bound to a non-loopback address with no auth token.");
        println!("           This process holds Kite credentials. Set DASHBOARD_AUTH_TOKEN.");
    }
    println!();
}
