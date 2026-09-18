//! Dashboard configuration, loaded from the environment.
//!
//! Security note: this process holds Kite credentials, so it binds to
//! localhost by default and secrets are never exposed to the browser.

use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    /// Bind address. Defaults to 127.0.0.1 - do not expose publicly without auth.
    pub bind: SocketAddr,
    /// Optional bearer token; when set, all /api routes require it.
    pub auth_token: Option<String>,
    /// Shared artifacts directory written by the Python analytics.
    pub artifacts_dir: PathBuf,
    /// Python project root (for launching backtest runs).
    pub python_dir: PathBuf,
    /// Python interpreter to invoke.
    pub python_bin: String,
    /// Static assets (HTML/JS/CSS).
    pub assets_dir: PathBuf,

    // --- Kite (read-only usage) ---
    pub kite_api_key: Option<String>,
    pub kite_api_secret: Option<String>,
    pub kite_access_token: Option<String>,
    /// Registered with Zerodha; the app serves the matching /kite/callback route.
    #[allow(dead_code)]
    pub kite_redirect_url: String,

    /// Optional webhook for alerts.
    pub alert_webhook: Option<String>,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        // Load .env from the repo root if present (ignore if missing).
        let _ = dotenvy::from_path("../.env");
        let _ = dotenvy::dotenv();

        let bind: SocketAddr = std::env::var("DASHBOARD_BIND")
            .unwrap_or_else(|_| "127.0.0.1:8080".to_string())
            .parse()?;

        let artifacts_dir = PathBuf::from(
            std::env::var("ARTIFACTS_DIR").unwrap_or_else(|_| "../artifacts".to_string()),
        );
        let python_dir = PathBuf::from(
            std::env::var("PYTHON_PROJECT_DIR").unwrap_or_else(|_| "../python".to_string()),
        );
        let assets_dir = PathBuf::from(
            std::env::var("ASSETS_DIR").unwrap_or_else(|_| "assets".to_string()),
        );

        Ok(Self {
            bind,
            auth_token: non_empty("DASHBOARD_AUTH_TOKEN"),
            artifacts_dir,
            python_dir,
            python_bin: std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python".to_string()),
            assets_dir,
            kite_api_key: non_empty("KITE_API_KEY"),
            kite_api_secret: non_empty("KITE_API_SECRET"),
            kite_access_token: non_empty("KITE_ACCESS_TOKEN"),
            kite_redirect_url: std::env::var("KITE_REDIRECT_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8080/kite/callback".to_string()),
            alert_webhook: non_empty("ALERT_WEBHOOK_URL"),
        })
    }

    pub fn kite_configured(&self) -> bool {
        self.kite_api_key.is_some() && self.kite_api_secret.is_some()
    }

    /// True when the server is reachable beyond the loopback interface.
    pub fn is_exposed(&self) -> bool {
        !self.bind.ip().is_loopback()
    }
}

fn non_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}
