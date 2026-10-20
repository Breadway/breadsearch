use std::{
    env, fs,
    io::{BufRead, BufReader, Write},
    os::unix::net::UnixStream,
    path::PathBuf,
};

use serde::{Deserialize, Serialize};

// ---- XDG path helpers -------------------------------------------------------

pub fn home_dir() -> PathBuf {
    PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
}

pub fn config_dir() -> PathBuf {
    env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".config"))
        .join("breadsearch")
}

pub fn state_dir() -> PathBuf {
    env::var("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".local/state"))
        .join("breadsearch")
}

pub fn cache_dir() -> PathBuf {
    env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir().join(".cache"))
        .join("breadsearch")
}

pub fn socket_path() -> PathBuf {
    env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join("breadmill.sock")
}

// ---- Config -----------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub index: IndexConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub power: PowerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    #[serde(default = "default_roots")]
    pub roots: Vec<String>,
    #[serde(default)]
    pub excludes: Vec<String>,
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,
    #[serde(default = "default_max_file_mb")]
    pub max_file_mb: f64,
}

fn default_roots() -> Vec<String> {
    let home = home_dir();
    vec![
        home.join("Documents").to_string_lossy().into_owned(),
        home.join("Projects").to_string_lossy().into_owned(),
        home.join(".config/breadpad").to_string_lossy().into_owned(),
    ]
}

fn default_extensions() -> Vec<String> {
    ["md", "txt", "org", "pdf", "odt", "docx"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn default_max_file_mb() -> f64 {
    10.0
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            roots: default_roots(),
            excludes: vec![],
            extensions: default_extensions(),
            max_file_mb: default_max_file_mb(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_snippet_len")]
    pub snippet_len: usize,
}

fn default_limit() -> usize {
    10
}

fn default_snippet_len() -> usize {
    200
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: default_limit(),
            snippet_len: default_snippet_len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default = "default_model_name")]
    pub name: String,
    #[serde(default = "default_dim")]
    pub dim: usize,
    /// Compute backend: "cpu", "npu" (VitisAI/XDNA), "rocm" (MIGraphX/AMD GPU),
    /// "cuda" (NVIDIA GPU), or "openvino" (Intel iGPU/dGPU).
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_model_name() -> String {
    "nomic-embed-text-v1.5".into()
}

fn default_dim() -> usize {
    768
}

fn default_backend() -> String {
    "cpu".into()
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: default_model_name(),
            dim: default_dim(),
            backend: default_backend(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConfig {
    /// Master on/off switch for indexing (embedding). When false, breadmill
    /// still serves queries over the existing index but never embeds new files.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Whether to keep embedding while running on battery. Embedding is the
    /// compute-heavy step (CPU/NPU/GPU forward pass), so this defaults to
    /// false to avoid draining battery; indexing resumes automatically once
    /// AC power is reconnected.
    #[serde(default)]
    pub run_on_battery: bool,
}

fn default_true() -> bool {
    true
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            run_on_battery: false,
        }
    }
}

impl Config {
    pub fn load() -> Self {
        let path = config_dir().join("config.toml");
        let content = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(e) => {
                eprintln!("breadsearch: could not read {}: {}", path.display(), e);
                return Self::default();
            }
        };
        match toml::from_str(&content) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("breadsearch: parse error in {}: {}", path.display(), e);
                Self::default()
            }
        }
    }
}

// ---- IPC types (newline-delimited JSON over Unix socket) --------------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Query {
        query: String,
        limit: usize,
    },
    Status,
    Reindex,
    /// Drop the loaded model session, freeing its backend allocations.
    /// Refused while queries are in flight. Unknown to older daemons, which
    /// answer a parse error: that is how a client tells them apart.
    UnloadModel,
    /// Load (or reload) the model in the background. Accepted even when a
    /// load is already running; readiness is polled through `Status`.
    LoadModel,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Hits { hits: Vec<Hit> },
    StatusInfo(StatusInfo),
    Ok,
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub title: String,
    pub path: String,
    pub snippet: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusInfo {
    pub indexed: usize,
    pub pending: usize,
    pub model_ready: bool,
    /// Queries currently inside an embedding call. Absent on older daemons.
    #[serde(default)]
    pub in_flight: usize,
    /// A background model load is running. Absent on older daemons.
    #[serde(default)]
    pub loading: bool,
    /// The model file size in bytes from the last load, if one happened.
    /// Absent on older daemons.
    #[serde(default)]
    pub model_bytes: Option<u64>,
    /// The daemon version that answered. Absent on older daemons, which is
    /// exactly how a client tells them apart from one with unload verbs.
    #[serde(default)]
    pub version: Option<String>,
    /// Total queries the daemon has served since it started. Absent on older daemons.
    #[serde(default)]
    pub queries_served: Option<u64>,
    #[serde(default)]
    pub model_error: Option<String>,
}

// ---- Socket client ----------------------------------------------------------

pub fn send_request(req: &Request) -> std::io::Result<Response> {
    let mut stream = UnixStream::connect(socket_path())?;

    let mut line = serde_json::to_string(req).map_err(std::io::Error::other)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    stream.flush()?;

    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response)?;

    serde_json::from_str(response.trim())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_from_older_daemon_defaults_lifecycle_fields() {
        let status: StatusInfo =
            serde_json::from_str(r#"{"indexed":2,"pending":0,"model_ready":true}"#).unwrap();
        assert_eq!(status.in_flight, 0);
        assert!(!status.loading);
        assert!(status.model_bytes.is_none());
        assert!(status.version.is_none());
        assert!(status.queries_served.is_none());
        assert!(status.model_error.is_none());
    }

    #[test]
    fn status_preserves_lifecycle_fields_and_model_errors() {
        let response: Response = serde_json::from_str(
            r#"{"type":"status_info","indexed":2,"pending":0,"model_ready":false,"in_flight":0,"loading":false,"model_bytes":400,"version":"0.3.3","queries_served":7,"model_error":"load failed"}"#,
        )
        .unwrap();
        let encoded = serde_json::to_value(response).unwrap();
        assert_eq!(encoded["model_error"], "load failed");
        assert_eq!(encoded["model_bytes"], 400);
        assert_eq!(encoded["queries_served"], 7);
        assert_eq!(encoded["version"], "0.3.3");
    }
}
