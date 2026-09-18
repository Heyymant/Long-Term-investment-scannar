//! Kite Connect REST client - READ ONLY.
//!
//! Deliberately exposes only read endpoints (profile, holdings, positions,
//! quotes, historical candles, orders/trades for the journal). There is no
//! order-placement method in this file, and there should not be one: the
//! dashboard is decision support, not an execution system.

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const BASE_URL: &str = "https://api.kite.trade";
const API_VERSION: &str = "3";

#[derive(Debug, Clone)]
pub struct KiteClient {
    api_key: String,
    api_secret: String,
    access_token: Option<String>,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct KiteResponse<T> {
    #[serde(default)]
    status: String,
    data: Option<T>,
    #[serde(default)]
    message: Option<String>,
}

// The full read-only Kite surface is implemented here; not every endpoint has
// a route yet (profile/orders/quote are available for the journal and future
// views). Nothing in this impl can place an order.
#[allow(dead_code)]
impl KiteClient {
    pub fn new(api_key: String, api_secret: String, access_token: Option<String>) -> Self {
        Self {
            api_key,
            api_secret,
            access_token,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn has_token(&self) -> bool {
        self.access_token.is_some()
    }

    pub fn set_access_token(&mut self, token: String) {
        self.access_token = Some(token);
    }

    pub fn login_url(&self) -> String {
        format!("https://kite.zerodha.com/connect/login?v={API_VERSION}&api_key={}", self.api_key)
    }

    /// Exchange a request_token for an access_token.
    ///
    /// Kite requires checksum = SHA256(api_key + request_token + api_secret).
    /// Tokens expire daily, so this runs once per trading session.
    pub async fn generate_session(&mut self, request_token: &str) -> Result<String> {
        let checksum = {
            let mut hasher = Sha256::new();
            hasher.update(self.api_key.as_bytes());
            hasher.update(request_token.as_bytes());
            hasher.update(self.api_secret.as_bytes());
            hex::encode(hasher.finalize())
        };

        let resp = self
            .http
            .post(format!("{BASE_URL}/session/token"))
            .header("X-Kite-Version", API_VERSION)
            .form(&[
                ("api_key", self.api_key.as_str()),
                ("request_token", request_token),
                ("checksum", &checksum),
            ])
            .send()
            .await
            .context("Kite session request failed")?;

        let body: KiteResponse<Value> = resp.json().await.context("invalid session response")?;
        if body.status != "success" {
            anyhow::bail!("Kite login failed: {}", body.message.unwrap_or_default());
        }

        let token = body
            .data
            .as_ref()
            .and_then(|d| d.get("access_token"))
            .and_then(|t| t.as_str())
            .context("no access_token in Kite response")?
            .to_string();

        self.access_token = Some(token.clone());
        Ok(token)
    }

    async fn get(&self, path: &str) -> Result<Value> {
        let token = self
            .access_token
            .as_ref()
            .context("no Kite access token - log in via /kite/login first")?;

        let resp = self
            .http
            .get(format!("{BASE_URL}{path}"))
            .header("X-Kite-Version", API_VERSION)
            .header("Authorization", format!("token {}:{}", self.api_key, token))
            .send()
            .await
            .with_context(|| format!("Kite GET {path} failed"))?;

        let status = resp.status();
        let body: KiteResponse<Value> = resp.json().await.context("invalid Kite response")?;

        if !status.is_success() || body.status != "success" {
            let msg = body.message.unwrap_or_else(|| status.to_string());
            if status.as_u16() == 403 {
                anyhow::bail!("Kite token expired or invalid ({msg}). Log in again - tokens last one day.");
            }
            anyhow::bail!("Kite error on {path}: {msg}");
        }
        body.data.context("empty Kite response")
    }

    pub async fn profile(&self) -> Result<Value> {
        self.get("/user/profile").await
    }

    pub async fn holdings(&self) -> Result<Value> {
        self.get("/portfolio/holdings").await
    }

    pub async fn positions(&self) -> Result<Value> {
        self.get("/portfolio/positions").await
    }

    /// Executed trades - the input for the execution journal.
    pub async fn trades(&self) -> Result<Value> {
        self.get("/trades").await
    }

    pub async fn orders(&self) -> Result<Value> {
        self.get("/orders").await
    }

    pub async fn quote(&self, instruments: &[String]) -> Result<Value> {
        if instruments.is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        let query = instruments
            .iter()
            .map(|i| format!("i={}", urlencode(i)))
            .collect::<Vec<_>>()
            .join("&");
        self.get(&format!("/quote?{query}")).await
    }

    pub async fn ltp(&self, instruments: &[String]) -> Result<Value> {
        if instruments.is_empty() {
            return Ok(Value::Object(Default::default()));
        }
        let query = instruments
            .iter()
            .map(|i| format!("i={}", urlencode(i)))
            .collect::<Vec<_>>()
            .join("&");
        self.get(&format!("/quote/ltp?{query}")).await
    }

    /// Websocket URL for the ticker (token is in the query string).
    pub fn ws_url(&self) -> Option<String> {
        let token = self.access_token.as_ref()?;
        Some(format!("wss://ws.kite.trade?api_key={}&access_token={token}", self.api_key))
    }
}

fn urlencode(s: &str) -> String {
    s.replace(' ', "%20").replace('&', "%26").replace('#', "%23")
}
