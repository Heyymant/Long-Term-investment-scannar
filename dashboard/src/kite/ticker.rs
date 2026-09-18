//! KiteTicker WebSocket client and binary packet parser.
//!
//! Kite streams quotes as a compact binary protocol rather than JSON:
//!
//!   [2 bytes] number of packets in this message
//!   for each packet:
//!     [2 bytes] packet length
//!     [N bytes] payload, big-endian int32 fields
//!
//! Payload sizes: 8 = LTP mode, 44 = quote mode (index), 184 = full mode.
//! Prices arrive as paise (1/100 rupee) for NSE equity, so we divide by 100.
//!
//! Ticks are relayed to browsers over our own WebSocket; the Kite access
//! token never leaves the server.

use std::sync::Arc;

use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, info, warn};

use crate::data::models::Tick;

/// Price divisor by exchange segment (segment = token & 0xFF).
fn price_divisor(instrument_token: u32) -> f64 {
    match instrument_token & 0xFF {
        3 => 10_000_000.0, // CDS
        6 => 1_000.0,      // BCD
        _ => 100.0,        // NSE/BSE equity and F&O: paise
    }
}

/// Parse one binary frame into ticks.
pub fn parse_binary_frame(data: &[u8]) -> Vec<Tick> {
    let mut ticks = Vec::new();
    if data.len() < 2 {
        return ticks;
    }

    let n_packets = i16::from_be_bytes([data[0], data[1]]) as usize;
    let mut offset = 2usize;

    for _ in 0..n_packets {
        if offset + 2 > data.len() {
            break;
        }
        let len = i16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if len == 0 || offset + len > data.len() {
            break;
        }
        if let Some(tick) = parse_packet(&data[offset..offset + len]) {
            ticks.push(tick);
        }
        offset += len;
    }
    ticks
}

fn parse_packet(p: &[u8]) -> Option<Tick> {
    if p.len() < 8 {
        return None;
    }
    let token = u32::from_be_bytes([p[0], p[1], p[2], p[3]]);
    let divisor = price_divisor(token);
    let last_price = be_i32(p, 4)? as f64 / divisor;

    // LTP mode stops here.
    if p.len() < 44 {
        return Some(Tick { instrument_token: token, last_price, volume: None, ohlc_close: None, change: None });
    }

    // Quote/full mode: volume at 16, close at 40 (index packets differ but
    // the fields we read are in the same place for equities).
    let volume = be_i32(p, 16).map(|v| v as u32);
    let close = be_i32(p, 40).map(|c| c as f64 / divisor);
    let change = close.and_then(|c| (c > 0.0).then(|| (last_price - c) / c * 100.0));

    Some(Tick { instrument_token: token, last_price, volume, ohlc_close: close, change })
}

fn be_i32(p: &[u8], at: usize) -> Option<i32> {
    (p.len() >= at + 4).then(|| i32::from_be_bytes([p[at], p[at + 1], p[at + 2], p[at + 3]]))
}

/// Connect to the Kite ticker and broadcast ticks to subscribers.
///
/// Reconnects with backoff; the stream only carries data during market hours.
pub async fn run_ticker(
    ws_url: String,
    instrument_tokens: Vec<u32>,
    tx: broadcast::Sender<Tick>,
    shutdown: Arc<tokio::sync::Notify>,
) -> Result<()> {
    let mut backoff = 1u64;

    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                info!("ticker shutting down");
                return Ok(());
            }
            result = connect_once(&ws_url, &instrument_tokens, &tx) => {
                match result {
                    Ok(()) => {
                        info!("ticker stream ended; reconnecting in {backoff}s");
                    }
                    Err(e) => {
                        warn!("ticker error: {e}; reconnecting in {backoff}s");
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
            }
        }
    }
}

async fn connect_once(
    ws_url: &str,
    tokens: &[u32],
    tx: &broadcast::Sender<Tick>,
) -> Result<()> {
    let (ws_stream, _) = tokio_tungstenite::connect_async(ws_url).await?;
    let (mut write, mut read) = ws_stream.split();
    info!("connected to Kite ticker ({} instruments)", tokens.len());

    // Kite caps subscriptions (~3000 tokens); chunk to stay well inside it.
    for chunk in tokens.chunks(500) {
        let subscribe = serde_json::json!({ "a": "subscribe", "v": chunk });
        write.send(Message::Text(subscribe.to_string())).await?;
        let mode = serde_json::json!({ "a": "mode", "v": ["quote", chunk] });
        write.send(Message::Text(mode.to_string())).await?;
    }

    while let Some(msg) = read.next().await {
        match msg? {
            Message::Binary(data) => {
                // A 1-byte payload is Kite's heartbeat.
                if data.len() <= 1 {
                    continue;
                }
                for tick in parse_binary_frame(&data) {
                    let _ = tx.send(tick); // ignore: no subscribers is fine
                }
            }
            Message::Text(text) => debug!("ticker message: {text}"),
            Message::Ping(p) => write.send(Message::Pong(p)).await?,
            Message::Close(_) => break,
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ltp_packet() {
        // 1 packet, 8 bytes: token=408065, ltp=123456 paise -> 1234.56
        let mut frame = vec![0u8, 1];
        frame.extend_from_slice(&8i16.to_be_bytes());
        frame.extend_from_slice(&408065u32.to_be_bytes());
        frame.extend_from_slice(&123456i32.to_be_bytes());

        let ticks = parse_binary_frame(&frame);
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].instrument_token, 408065);
        assert!((ticks[0].last_price - 1234.56).abs() < 1e-9);
    }

    #[test]
    fn ignores_truncated_frame() {
        let frame = vec![0u8, 1, 0, 40, 1, 2, 3];
        assert!(parse_binary_frame(&frame).is_empty());
    }
}
