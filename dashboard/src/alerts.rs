//! Optional outbound alerts (regime flips, new SUE events, live drift).
//!
//! Fire-and-forget webhooks. Alerts are informational: the system still never
//! acts on them, you do.

use serde::Serialize;
use tracing::{debug, warn};

/// Not every alert kind is wired to a trigger yet; they define the contract.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertKind {
    RegimeFlip,
    NewEventSignal,
    RebalanceDue,
    LiveDrift,
    DataHealth,
}

#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub kind: AlertKind,
    pub title: String,
    pub body: String,
    pub timestamp: i64,
}

impl Alert {
    pub fn new(kind: AlertKind, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            body: body.into(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}

#[derive(Clone)]
pub struct Alerter {
    webhook: Option<String>,
    http: reqwest::Client,
}

impl Alerter {
    pub fn new(webhook: Option<String>) -> Self {
        Self {
            webhook,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.webhook.is_some()
    }

    /// Post an alert. Failures are logged, never propagated - a broken webhook
    /// must not take down the dashboard.
    pub async fn send(&self, alert: Alert) {
        let Some(url) = &self.webhook else {
            debug!("alert (no webhook configured): {} - {}", alert.title, alert.body);
            return;
        };

        let payload = serde_json::json!({
            "text": format!("[{}] {}\n{}", kind_label(alert.kind), alert.title, alert.body),
            "alert": alert,
        });

        match self.http.post(url).json(&payload).send().await {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => warn!("alert webhook returned {}", resp.status()),
            Err(e) => warn!("alert webhook failed: {e}"),
        }
    }
}

fn kind_label(kind: AlertKind) -> &'static str {
    match kind {
        AlertKind::RegimeFlip => "REGIME",
        AlertKind::NewEventSignal => "EVENT",
        AlertKind::RebalanceDue => "REBALANCE",
        AlertKind::LiveDrift => "DRIFT",
        AlertKind::DataHealth => "DATA",
    }
}
