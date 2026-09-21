//! Artifact readers with hot-reload.
//!
//! The Python analytics writes a compact JSON sibling next to every table
//! (`rankings.parquet` -> `rankings.json`), so the dashboard reads a single
//! stable format and needs no dataframe library. Parsed results are cached and
//! invalidated by a filesystem watcher, so re-running a backtest refreshes the
//! UI without a restart.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use serde_json::Value;
use tracing::{debug, info, warn};

use super::models::{SeriesData, TableResponse, SUPPORTED_SCHEMA};

pub struct ArtifactStore {
    root: PathBuf,
    cache: RwLock<HashMap<String, CachedEntry>>,
}

struct CachedEntry {
    value: Value,
    modified: SystemTime,
}

impl ArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into(), cache: RwLock::new(HashMap::new()) }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The run the dashboard should display (from `latest.json`).
    pub fn latest_run_id(&self) -> Option<String> {
        let text = std::fs::read_to_string(self.root.join("latest.json")).ok()?;
        let json: Value = serde_json::from_str(&text).ok()?;
        json.get("run_id")?.as_str().map(str::to_string)
    }

    pub fn run_dir(&self, run_id: Option<&str>) -> Option<PathBuf> {
        let id = match run_id {
            Some(r) => r.to_string(),
            None => self.latest_run_id()?,
        };
        let dir = self.root.join("runs").join(&id);
        dir.is_dir().then_some(dir)
    }

    pub fn list_runs(&self) -> Vec<String> {
        let Ok(entries) = std::fs::read_dir(self.root.join("runs")) else {
            return vec![];
        };
        let mut out: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        out.sort_by(|a, b| b.cmp(a)); // newest first (ids are timestamp-prefixed)
        out
    }

    // -- JSON ---------------------------------------------------------------
    pub fn read_json(&self, run_id: Option<&str>, name: &str) -> Result<Value> {
        let dir = self
            .run_dir(run_id)
            .context("no run directory found - run a backtest first")?;
        self.read_json_at(&dir.join(name))
    }

    /// Look in the run directory first, then fall back to the artifacts root
    /// (some files, like rebalance_orders, are written outside a run).
    pub fn read_json_any(&self, run_id: Option<&str>, name: &str) -> Result<Value> {
        if let Some(dir) = self.run_dir(run_id) {
            let path = dir.join(name);
            if path.exists() {
                return self.read_json_at(&path);
            }
        }
        self.read_json_at(&self.root.join(name))
    }

    fn read_json_at(&self, path: &Path) -> Result<Value> {
        let key = path.to_string_lossy().to_string();
        let modified = std::fs::metadata(path)
            .with_context(|| format!("artifact not found: {}", path.display()))?
            .modified()?;

        if let Some(hit) = self.cache.read().ok().and_then(|c| {
            c.get(&key)
                .and_then(|e| (e.modified == modified).then(|| e.value.clone()))
        }) {
            return Ok(hit);
        }

        let text = std::fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("invalid JSON in {}", path.display()))?;
        check_schema(&value, path);

        if let Ok(mut cache) = self.cache.write() {
            cache.insert(key, CachedEntry { value: value.clone(), modified });
        }
        Ok(value)
    }

    // -- Tables -------------------------------------------------------------
    /// Read a table artifact. Accepts the parquet/csv name for convenience and
    /// transparently loads the `.json` sibling.
    pub fn read_table(&self, run_id: Option<&str>, name: &str) -> Result<TableResponse> {
        let json_name = to_json_name(name);
        let value = self.read_json_any(run_id, &json_name)?;
        parse_table(&value)
            .with_context(|| format!("{json_name} is not a table artifact"))
    }

    pub fn invalidate(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        debug!("artifact cache cleared");
    }
}

fn to_json_name(name: &str) -> String {
    match name.rsplit_once('.') {
        Some((stem, ext)) if ext != "json" => format!("{stem}.json"),
        _ => name.to_string(),
    }
}

fn check_schema(value: &Value, path: &Path) {
    if let Some(v) = value.get("schema_version").and_then(Value::as_str) {
        if v != SUPPORTED_SCHEMA {
            warn!(
                "schema mismatch in {}: artifact is {v}, dashboard expects {SUPPORTED_SCHEMA}. \
                 Regenerate artifacts or update the dashboard.",
                path.display()
            );
        }
    }
}

/// `{columns: [...], rows: [[...]]}` -> TableResponse.
pub fn parse_table(value: &Value) -> Option<TableResponse> {
    let columns: Vec<String> = value
        .get("columns")?
        .as_array()?
        .iter()
        .map(|c| c.as_str().unwrap_or_default().to_string())
        .collect();

    let rows: Vec<Vec<Value>> = value
        .get("rows")?
        .as_array()?
        .iter()
        .filter_map(|r| r.as_array().cloned())
        .collect();

    Some(TableResponse { n_rows: rows.len(), columns, rows, note: None })
}

/// Turn a table artifact into the columnar arrays the D3 charts expect.
pub fn table_to_series(table: &TableResponse, time_col: &str) -> Option<SeriesData> {
    let t_idx = table.columns.iter().position(|c| c == time_col)?;

    let t: Vec<i64> = table
        .rows
        .iter()
        .map(|r| r.get(t_idx).and_then(parse_timestamp).unwrap_or(0))
        .collect();

    let mut series = HashMap::new();
    for (i, name) in table.columns.iter().enumerate() {
        if i == t_idx {
            continue;
        }
        let values: Vec<Option<f64>> = table
            .rows
            .iter()
            .map(|r| r.get(i).and_then(Value::as_f64).filter(|v| v.is_finite()))
            .collect();
        // Skip columns that carry no numeric data at all (e.g. labels).
        if values.iter().any(Option::is_some) {
            series.insert(name.clone(), values);
        }
    }
    Some(SeriesData { t, series })
}

fn parse_timestamp(v: &Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        // Heuristic: values this large are already epoch millis.
        return Some(if n > 100_000_000_000 { n / 1000 } else { n });
    }
    let s = v.as_str()?;
    let date_part = &s[..10.min(s.len())];
    let date = chrono::NaiveDate::parse_from_str(date_part, "%Y-%m-%d").ok()?;
    Some(date.and_hms_opt(0, 0, 0)?.and_utc().timestamp())
}

/// Read one numeric column out of a table.
pub fn column_f64(table: &TableResponse, name: &str) -> Vec<Option<f64>> {
    let Some(idx) = table.columns.iter().position(|c| c == name) else {
        return vec![];
    };
    table
        .rows
        .iter()
        .map(|r| r.get(idx).and_then(Value::as_f64))
        .collect()
}

/// Read one string column out of a table.
pub fn column_str(table: &TableResponse, name: &str) -> Vec<Option<String>> {
    let Some(idx) = table.columns.iter().position(|c| c == name) else {
        return vec![];
    };
    table
        .rows
        .iter()
        .map(|r| r.get(idx).and_then(Value::as_str).map(str::to_string))
        .collect()
}

// --------------------------------------------------------------------------
// Hot reload
// --------------------------------------------------------------------------
pub fn spawn_watcher(store: Arc<ArtifactStore>) -> Result<()> {
    use notify::{EventKind, RecursiveMode, Watcher};

    let root = store.root().to_path_buf();
    std::fs::create_dir_all(&root).ok();

    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = match notify::recommended_watcher(tx) {
            Ok(w) => w,
            Err(e) => {
                warn!("could not start artifact watcher: {e}");
                return;
            }
        };
        if let Err(e) = watcher.watch(&root, RecursiveMode::Recursive) {
            warn!("could not watch {}: {e}", root.display());
            return;
        }
        info!("watching {} for artifact changes", root.display());

        // Debounce: a single backtest writes many files in quick succession.
        let mut pending = false;
        loop {
            match rx.recv_timeout(Duration::from_millis(400)) {
                Ok(Ok(event)) => {
                    if matches!(
                        event.kind,
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                    ) {
                        pending = true;
                    }
                }
                Ok(Err(e)) => debug!("watch error: {e}"),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    if pending {
                        store.invalidate();
                        pending = false;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn maps_parquet_names_to_json_siblings() {
        assert_eq!(to_json_name("rankings.parquet"), "rankings.json");
        assert_eq!(to_json_name("rebalance_orders.csv"), "rebalance_orders.json");
        assert_eq!(to_json_name("risk.json"), "risk.json");
    }

    #[test]
    fn parses_a_table_artifact() {
        let v = json!({
            "columns": ["date", "equity"],
            "rows": [["2024-01-01", 100.0], ["2024-01-02", 101.5]],
        });
        let t = parse_table(&v).unwrap();
        assert_eq!(t.columns, vec!["date", "equity"]);
        assert_eq!(t.n_rows, 2);

        let s = table_to_series(&t, "date").unwrap();
        assert_eq!(s.t.len(), 2);
        assert_eq!(s.series["equity"][1], Some(101.5));
    }
}
