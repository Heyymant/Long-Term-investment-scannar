//! Shared application state.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};

use crate::alerts::Alerter;
use crate::config::Config;
use crate::data::artifacts::ArtifactStore;
use crate::data::models::{JobUpdate, Tick};
use crate::jobs::JobManager;
use crate::kite::rest::KiteClient;

pub type SharedState = Arc<AppState>;

pub struct AppState {
    pub config: Config,
    pub artifacts: Arc<ArtifactStore>,
    /// Kite client behind a lock because the access token is refreshed daily.
    pub kite: RwLock<Option<KiteClient>>,
    pub jobs: JobManager,
    /// Live ticks fanned out to browser WebSocket clients.
    pub ticks: broadcast::Sender<Tick>,
    /// Backtest/optimization job progress.
    pub job_events: broadcast::Sender<JobUpdate>,
    /// Optional outbound notifications (regime flips, drift, data issues).
    pub alerter: Alerter,
    /// instrument_token → ticker, filled from Kite quote so the ticker can
    /// attach symbols the way Databento sends SymbolMappingMsg.
    pub symbol_directory: Arc<RwLock<HashMap<u32, String>>>,
    /// Extra tokens to subscribe (screener universe, not just the target book).
    pub watch_tokens: Arc<RwLock<Vec<u32>>>,
}

impl AppState {
    pub fn new(config: Config) -> Arc<Self> {
        let artifacts = Arc::new(ArtifactStore::new(config.artifacts_dir.clone()));
        let (ticks, _) = broadcast::channel(2048);
        let (job_events, _) = broadcast::channel(256);

        let kite = if config.kite_configured() {
            Some(KiteClient::new(
                config.kite_api_key.clone().unwrap_or_default(),
                config.kite_api_secret.clone().unwrap_or_default(),
                config.kite_access_token.clone(),
            ))
        } else {
            None
        };

        Arc::new(Self {
            jobs: JobManager::new(
                config.python_bin.clone(),
                config.python_dir.clone(),
                job_events.clone(),
            ),
            alerter: Alerter::new(config.alert_webhook.clone()),
            config,
            artifacts,
            kite: RwLock::new(kite),
            ticks,
            job_events,
            symbol_directory: Arc::new(RwLock::new(HashMap::new())),
            watch_tokens: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Kite client, or a clear error explaining what is missing.
    pub async fn kite_client(&self) -> Result<KiteClient, String> {
        let guard = self.kite.read().await;
        match guard.as_ref() {
            None => Err(
                "Kite is not configured. Set KITE_API_KEY and KITE_API_SECRET in .env.".to_string()
            ),
            Some(c) if !c.has_token() => Err(
                "No Kite access token for today. Visit /kite/login to authenticate \
                 (tokens expire daily)."
                    .to_string(),
            ),
            Some(c) => Ok(c.clone()),
        }
    }
}
