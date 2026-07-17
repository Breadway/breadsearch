use std::path::{Path, PathBuf};

use bread_onnx::embedding::EmbeddingSession;
use bread_onnx::Provider;

const DOCUMENT_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";

/// Hard token cap for nomic-embed-text-v1.5 (8192-token context window).
/// Sequences longer than this are truncated before ONNX inference to prevent
/// quadratic attention memory blowup.
pub const MAX_SEQ_LEN: usize = 8192;

pub enum Backend {
    Cpu,
    /// AMD XDNA NPU via the VitisAI ONNX Runtime execution provider.
    /// `cache_dir` is used to store the compiled NPU model between runs.
    Npu { cache_dir: PathBuf },
    /// AMD iGPU via the MIGraphX ONNX Runtime execution provider (ROCm-backed).
    /// Distro ROCm ONNX Runtime builds (e.g. Arch's onnxruntime-rocm) are
    /// commonly compiled with `--use_migraphx`, not `--use_rocm`, so this
    /// targets `MIGraphXExecutionProvider` rather than the classic
    /// `ROCMExecutionProvider`.
    Rocm,
    /// NVIDIA GPU via the CUDA ONNX Runtime execution provider.
    Cuda,
    /// Intel iGPU/dGPU (Arc) via the OpenVINO ONNX Runtime execution
    /// provider, requesting device_type "GPU". `cache_dir` stores OpenVINO's
    /// compiled-model blobs between runs (its own `with_cache_dir`, not an
    /// env var — unlike MIGraphX this doesn't need a workaround).
    OpenVino { cache_dir: PathBuf },
}

pub struct OrtEmbedder {
    inner: EmbeddingSession,
}

impl OrtEmbedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path, dim: usize, backend: Backend) -> Result<Self, String> {
        let provider = to_provider(backend)?;
        let inner = EmbeddingSession::load(model_path, tokenizer_path, dim, MAX_SEQ_LEN, &[provider])
            .map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    pub fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_with_prefix(text, DOCUMENT_PREFIX)
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_with_prefix(text, QUERY_PREFIX)
    }

    fn embed_with_prefix(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>, String> {
        let input = format!("{}{}", prefix, text);
        self.inner.embed(&input).map_err(|e| e.to_string())
    }
}

// ---- Backend -> bread_onnx::Provider ----------------------------------------
//
// `bread_onnx::session::build_session` (via `EmbeddingSession::load`) is the
// shared session-builder + EP-fallback + loud-logging code every EP branch
// below used to hand-roll separately (`configure_eps`/`npu_session`/
// `rocm_session`/`cuda_session`/`openvino_session`). What's genuinely
// specific to this crate — its cargo feature gates (npu/rocm/cuda/openvino),
// and NPU's `vaip_config.json` discovery — stays here.

fn to_provider(backend: Backend) -> Result<Provider, String> {
    match backend {
        Backend::Cpu => Ok(Provider::Cpu),
        Backend::Npu { cache_dir } => npu_provider(cache_dir),
        Backend::Rocm => rocm_provider(),
        Backend::Cuda => cuda_provider(),
        Backend::OpenVino { cache_dir } => Ok(Provider::OpenVino { device_type: "GPU".to_string(), cache_dir }),
    }
}

#[cfg(feature = "npu")]
fn npu_provider(cache_dir: PathBuf) -> Result<Provider, String> {
    let vaip_config = find_vaip_config()?;
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        eprintln!(
            "breadmill: hint — set ORT_DYLIB_PATH to the Ryzen AI SDK ORT, e.g.:\n  \
             ORT_DYLIB_PATH=~/.local/share/ryzen-ai-1.7.1/lib/libonnxruntime.so"
        );
    }
    Ok(Provider::Vitis {
        config_file: vaip_config,
        cache_dir: cache_dir.join("npu"),
        cache_key: "nomic-embed-text-v1.5".to_string(),
    })
}

#[cfg(not(feature = "npu"))]
fn npu_provider(_cache_dir: PathBuf) -> Result<Provider, String> {
    eprintln!("breadmill: NPU backend requested but not compiled in (rebuild with --features npu); using CPU");
    Ok(Provider::Cpu)
}

#[cfg(feature = "rocm")]
fn rocm_provider() -> Result<Provider, String> {
    Ok(Provider::MiGraphX { device_id: 0 })
}

#[cfg(not(feature = "rocm"))]
fn rocm_provider() -> Result<Provider, String> {
    eprintln!("breadmill: ROCm backend requested but not compiled in (rebuild with --features rocm); using CPU");
    Ok(Provider::Cpu)
}

#[cfg(feature = "cuda")]
fn cuda_provider() -> Result<Provider, String> {
    Ok(Provider::Cuda { device_id: 0 })
}

#[cfg(not(feature = "cuda"))]
fn cuda_provider() -> Result<Provider, String> {
    eprintln!("breadmill: CUDA backend requested but not compiled in (rebuild with --features cuda); using CPU");
    Ok(Provider::Cpu)
}

/// Locate the VitisAI EP config file required by the AMD Ryzen AI SDK.
///
/// Search order:
/// 1. `VAIP_CONFIG` environment variable
/// 2. `~/.config/breadsearch/vaip_config.json`
/// 3. `/etc/vaip_config.json`
/// 4. `/opt/xilinx/vaip_config.json`
#[cfg(feature = "npu")]
fn find_vaip_config() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("VAIP_CONFIG") {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!("VAIP_CONFIG={p} does not exist"));
    }

    let user_path = breadsearch_shared::config_dir().join("vaip_config.json");
    if user_path.exists() {
        return Ok(user_path);
    }

    // Standard system / SDK paths (checked in priority order)
    let home = std::env::var("HOME").unwrap_or_default();
    let sdk_paths = [
        format!("{home}/.local/share/ryzen-ai-1.7.1/voe-4.0-linux_x86_64/vaip_config.json"),
        "/etc/vaip_config.json".into(),
        "/opt/xilinx/vaip_config.json".into(),
    ];
    for p in &sdk_paths {
        let path = Path::new(p.as_str());
        if path.exists() {
            return Ok(path.to_path_buf());
        }
    }

    Err(
        "vaip_config.json not found; set VAIP_CONFIG=/path/to/vaip_config.json, \
         copy to ~/.config/breadsearch/vaip_config.json, or install the AMD Ryzen AI SDK"
            .into(),
    )
}
