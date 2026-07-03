use std::path::{Path, PathBuf};

use ort::{
    session::{Session, builder::{GraphOptimizationLevel, SessionBuilder}},
    value::Tensor,
};
use tokenizers::Tokenizer;

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
}

pub struct OrtEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    dim: usize,
}

impl OrtEmbedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path, dim: usize, backend: Backend) -> Result<Self, String> {
        let builder = Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| e.to_string())?;

        let mut builder = configure_eps(builder, &backend)?;
        let session = builder.commit_from_file(model_path).map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;

        Ok(Self { session, tokenizer, dim })
    }

    pub fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_with_prefix(text, DOCUMENT_PREFIX)
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_with_prefix(text, QUERY_PREFIX)
    }

    fn embed_with_prefix(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>, String> {
        let input = format!("{}{}", prefix, text);

        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(|e| e.to_string())?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mut type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&x| x as i64)
            .collect();

        if ids.len() > MAX_SEQ_LEN {
            eprintln!(
                "breadmill: truncating {} tokens to {} (chunk too large)",
                ids.len(),
                MAX_SEQ_LEN
            );
            ids.truncate(MAX_SEQ_LEN);
            mask.truncate(MAX_SEQ_LEN);
            type_ids.truncate(MAX_SEQ_LEN);
        }

        let seq_len = ids.len() as i64;

        let id_tensor =
            Tensor::<i64>::from_array((vec![1i64, seq_len], ids.clone())).map_err(|e| e.to_string())?;
        let mask_tensor =
            Tensor::<i64>::from_array((vec![1i64, seq_len], mask.clone())).map_err(|e| e.to_string())?;
        let type_tensor =
            Tensor::<i64>::from_array((vec![1i64, seq_len], type_ids)).map_err(|e| e.to_string())?;

        let outputs = self
            .session
            .run(ort::inputs! {
                "input_ids" => id_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => type_tensor,
            })
            .map_err(|e| e.to_string())?;

        // last_hidden_state: shape [1, seq_len, dim]
        let (shape, data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        let actual_seq = shape[1] as usize;
        let actual_dim = shape[2] as usize;

        // Mean-pool over non-padding positions. Some execution providers (e.g.
        // MIGraphX) pad the output sequence dimension for kernel efficiency, so
        // actual_seq can exceed mask.len() — only positions covered by our own
        // attention mask are meaningful, so cap the loop at whichever is shorter.
        let mut result = vec![0.0f32; actual_dim];
        let mut count = 0usize;

        for t in 0..actual_seq.min(mask.len()) {
            if mask[t] > 0 {
                for d in 0..actual_dim {
                    result[d] += data[t * actual_dim + d];
                }
                count += 1;
            }
        }

        if count > 0 {
            for x in &mut result {
                *x /= count as f32;
            }
        }

        l2_normalize(&mut result);

        // Clamp/pad to configured dim
        result.truncate(self.dim);
        while result.len() < self.dim {
            result.push(0.0);
        }

        Ok(result)
    }
}

fn l2_normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---- Execution provider selection -------------------------------------------

fn configure_eps(builder: SessionBuilder, backend: &Backend) -> Result<SessionBuilder, String> {
    match backend {
        Backend::Cpu => Ok(builder),
        Backend::Npu { cache_dir } => npu_session(builder, cache_dir),
        Backend::Rocm => rocm_session(builder),
        Backend::Cuda => cuda_session(builder),
    }
}

#[cfg(feature = "npu")]
fn npu_session(builder: SessionBuilder, cache_dir: &Path) -> Result<SessionBuilder, String> {
    let vitis_ep = build_vitis_ep(cache_dir)?;
    eprintln!("breadmill: using NPU (VitisAI) execution provider");
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        eprintln!(
            "breadmill: hint — set ORT_DYLIB_PATH to the Ryzen AI SDK ORT, e.g.:\n  \
             ORT_DYLIB_PATH=~/.local/share/ryzen-ai-1.7.1/lib/libonnxruntime.so"
        );
    }
    builder
        .with_execution_providers([vitis_ep, ort::ep::CPU::default().build()])
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "npu"))]
fn npu_session(builder: SessionBuilder, _cache_dir: &Path) -> Result<SessionBuilder, String> {
    eprintln!("breadmill: NPU backend requested but not compiled in (rebuild with --features npu); using CPU");
    Ok(builder)
}

// ---- VitisAI EP (NPU) -------------------------------------------------------

#[cfg(feature = "npu")]
fn build_vitis_ep(cache_dir: &Path) -> Result<ort::ep::ExecutionProviderDispatch, String> {
    let vaip_config = find_vaip_config()?;
    let npu_cache = cache_dir.join("npu");
    std::fs::create_dir_all(&npu_cache).map_err(|e| e.to_string())?;
    eprintln!("breadmill: vaip_config: {}", vaip_config.display());
    eprintln!("breadmill: NPU model cache: {}", npu_cache.display());
    Ok(ort::ep::Vitis::default()
        .with_config_file(vaip_config.to_string_lossy())
        .with_cache_dir(npu_cache.to_string_lossy())
        .with_cache_key("nomic-embed-text-v1.5")
        .build())
}

// ---- MIGraphX EP (AMD iGPU, ROCm-backed) -------------------------------------

#[cfg(feature = "rocm")]
fn rocm_session(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    eprintln!("breadmill: using MIGraphX execution provider (device 0)");
    eprintln!(
        "breadmill: note — check the log line above/below for \"Successfully registered \
         `MIGraphXExecutionProvider`\"; if it's missing, the ONNX Runtime in use wasn't built \
         with MIGraphX support and inference silently fell back to CPU"
    );
    builder
        .with_execution_providers([
            ort::ep::MIGraphX::default().with_device_id(0).build(),
            ort::ep::CPU::default().build(),
        ])
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "rocm"))]
fn rocm_session(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    eprintln!("breadmill: ROCm backend requested but not compiled in (rebuild with --features rocm); using CPU");
    Ok(builder)
}

// ---- CUDA EP (NVIDIA GPU) ----------------------------------------------------

#[cfg(feature = "cuda")]
fn cuda_session(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    eprintln!("breadmill: using CUDA execution provider (device 0)");
    eprintln!(
        "breadmill: note — check the log line above/below for \"Successfully registered \
         `CUDAExecutionProvider`\"; if it's missing, the ONNX Runtime in use wasn't built \
         with CUDA support and inference silently fell back to CPU"
    );
    builder
        .with_execution_providers([
            ort::ep::CUDA::default().with_device_id(0).build(),
            ort::ep::CPU::default().build(),
        ])
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "cuda"))]
fn cuda_session(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    eprintln!("breadmill: CUDA backend requested but not compiled in (rebuild with --features cuda); using CPU");
    Ok(builder)
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
