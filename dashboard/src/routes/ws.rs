//! WebSocket endpoints: live ticks and job progress.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use tracing::debug;

use crate::state::SharedState;

/// GET /ws/ticks - relay live Kite ticks to the browser.
///
/// The browser never sees the Kite token; this server holds the upstream
/// connection and fans out ticks.
pub async fn ticks(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ticks(socket, state))
}

async fn handle_ticks(mut socket: WebSocket, state: SharedState) {
    let mut rx = state.ticks.subscribe();

    if socket
        .send(Message::Text(
            serde_json::json!({ "type": "connected", "stream": "ticks" }).to_string(),
        ))
        .await
        .is_err()
    {
        return;
    }

    loop {
        tokio::select! {
            received = rx.recv() => match received {
                Ok(tick) => {
                    let payload = match serde_json::to_string(&tick) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    if socket.send(Message::Text(payload)).await.is_err() {
                        break;
                    }
                }
                // A slow client can lag behind; drop the backlog rather than
                // stalling the broadcast for everyone else.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    debug!("tick client lagged by {n} messages");
                    continue;
                }
                Err(_) => break,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            }
        }
    }
}

/// GET /ws/jobs - stream backtest/optimization progress.
pub async fn jobs(ws: WebSocketUpgrade, State(state): State<SharedState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_jobs(socket, state))
}

async fn handle_jobs(mut socket: WebSocket, state: SharedState) {
    let mut rx = state.job_events.subscribe();

    // Send current job state so a reconnecting client is not left blank.
    let snapshot = serde_json::json!({ "type": "snapshot", "jobs": state.jobs.list().await });
    if socket.send(Message::Text(snapshot.to_string())).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            received = rx.recv() => match received {
                Ok(update) => {
                    let payload = match serde_json::to_string(&update) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    if socket.send(Message::Text(payload)).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                _ => {}
            }
        }
    }
}
