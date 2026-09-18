//! Launch and track Python analytics runs.
//!
//! The dashboard composes a strategy design, writes it to disk, and shells out
//! to the same CLI a human would use. Progress is streamed to the browser over
//! `/ws/jobs`. These are local analytics jobs only - nothing touches the broker.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;

use chrono::Utc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{broadcast, RwLock};
use tracing::{error, info};

use crate::data::models::{JobStatus, JobUpdate};

#[derive(Clone)]
pub struct JobManager {
    python_bin: String,
    python_dir: PathBuf,
    events: broadcast::Sender<JobUpdate>,
    jobs: Arc<RwLock<HashMap<String, JobUpdate>>>,
}

impl JobManager {
    pub fn new(python_bin: String, python_dir: PathBuf, events: broadcast::Sender<JobUpdate>) -> Self {
        Self {
            python_bin,
            python_dir,
            events,
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn list(&self) -> Vec<JobUpdate> {
        self.jobs.read().await.values().cloned().collect()
    }

    #[allow(dead_code)]
    pub async fn get(&self, job_id: &str) -> Option<JobUpdate> {
        self.jobs.read().await.get(job_id).cloned()
    }

    /// Spawn a backtest. Returns the job id immediately; progress arrives on
    /// the `/ws/jobs` stream.
    pub async fn run_backtest(&self, args: Vec<String>) -> String {
        let job_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        self.spawn(job_id.clone(), "scripts/run_backtest.py", args).await;
        job_id
    }

    pub async fn run_optimization(&self, args: Vec<String>) -> String {
        let job_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        self.spawn(job_id.clone(), "scripts/run_optimization.py", args).await;
        job_id
    }

    pub async fn run_validation(&self, args: Vec<String>) -> String {
        let job_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        self.spawn(job_id.clone(), "scripts/run_validation.py", args).await;
        job_id
    }

    pub async fn run_rebalance(&self, args: Vec<String>) -> String {
        let job_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        self.spawn(job_id.clone(), "scripts/generate_rebalance.py", args).await;
        job_id
    }

    async fn spawn(&self, job_id: String, script: &str, args: Vec<String>) {
        let update = JobUpdate {
            job_id: job_id.clone(),
            status: JobStatus::Queued,
            message: format!("queued {script}"),
            run_id: None,
            timestamp: Utc::now().timestamp(),
        };
        self.record(update).await;

        let python_bin = self.python_bin.clone();
        let python_dir = self.python_dir.clone();
        let script = script.to_string();
        let events = self.events.clone();
        let jobs = self.jobs.clone();

        tokio::spawn(async move {
            let emit = |status: JobStatus, message: String, run_id: Option<String>| {
                let u = JobUpdate {
                    job_id: job_id.clone(),
                    status,
                    message,
                    run_id,
                    timestamp: Utc::now().timestamp(),
                };
                let _ = events.send(u.clone());
                let jobs = jobs.clone();
                tokio::spawn(async move {
                    jobs.write().await.insert(u.job_id.clone(), u);
                });
            };

            emit(JobStatus::Running, format!("running {script}"), None);
            info!("job {job_id}: {python_bin} {script} {}", args.join(" "));

            let child = Command::new(&python_bin)
                .arg(&script)
                .args(&args)
                .current_dir(&python_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn();

            let mut child = match child {
                Ok(c) => c,
                Err(e) => {
                    error!("job {job_id} failed to start: {e}");
                    emit(JobStatus::Failed, format!("could not start python: {e}"), None);
                    return;
                }
            };

            let mut run_id = None;
            if let Some(stdout) = child.stdout.take() {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    // The CLI prints "RESULTS  (run <id>)"; capture it so the
                    // UI can jump straight to the new run's artifacts.
                    if let Some(id) = extract_run_id(&line) {
                        run_id = Some(id);
                    }
                    if !line.trim().is_empty() {
                        emit(JobStatus::Running, line.trim().to_string(), run_id.clone());
                    }
                }
            }

            let mut stderr_tail = String::new();
            if let Some(stderr) = child.stderr.take() {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if line.contains("ERROR") || line.contains("Traceback") {
                        stderr_tail.push_str(&line);
                        stderr_tail.push('\n');
                    }
                }
            }

            match child.wait().await {
                Ok(status) if status.success() => {
                    emit(JobStatus::Completed, "completed".to_string(), run_id);
                }
                Ok(status) => {
                    let msg = if stderr_tail.is_empty() {
                        format!("exited with status {status}")
                    } else {
                        stderr_tail.lines().last().unwrap_or("failed").to_string()
                    };
                    emit(JobStatus::Failed, msg, run_id);
                }
                Err(e) => emit(JobStatus::Failed, e.to_string(), run_id),
            }
        });
    }

    async fn record(&self, update: JobUpdate) {
        let _ = self.events.send(update.clone());
        self.jobs.write().await.insert(update.job_id.clone(), update);
    }
}

fn extract_run_id(line: &str) -> Option<String> {
    let start = line.find("(run ")? + 5;
    let rest = &line[start..];
    let end = rest.find(')')?;
    Some(rest[..end].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::extract_run_id;

    #[test]
    fn extracts_run_id_from_cli_output() {
        assert_eq!(
            extract_run_id("RESULTS  (run 20260919-012904-97b6d9)").as_deref(),
            Some("20260919-012904-97b6d9")
        );
        assert_eq!(extract_run_id("no id here"), None);
    }
}
