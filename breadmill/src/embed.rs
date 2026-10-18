use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

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
    Npu {
        cache_dir: PathBuf,
    },
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
    /// env var; unlike MIGraphX this doesn't need a workaround).
    OpenVino {
        cache_dir: PathBuf,
    },
}
pub struct OrtEmbedder {
    inner: EmbeddingSession,
}

impl OrtEmbedder {
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        dim: usize,
        backend: Backend,
    ) -> Result<Self, String> {
        ensure_ort_runtime()?;
        let provider = to_provider(backend)?;
        let inner =
            EmbeddingSession::load(model_path, tokenizer_path, dim, MAX_SEQ_LEN, &[provider])
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

/// Resolve and load the ONNX Runtime library before any `ort` API is used.
///
/// Order: `ORT_DYLIB_PATH` if set (explicit override, e.g. a GPU-enabled
/// runtime), otherwise the CPU runtime build.rs provisioned into the build
/// output. A missing or unloadable library is returned as an error. ort's
/// own loader panics on this, which would kill the loader thread.
fn ensure_ort_runtime() -> Result<(), String> {
    let override_path = std::env::var_os("ORT_DYLIB_PATH").filter(|p| !p.is_empty());
    let (path, bundled) = if let Some(path) = override_path {
        (PathBuf::from(path), false)
    } else {
        let cache = breadsearch_shared::cache_dir().join("runtime");
        install_bundled_runtime(&cache)?;
        (cache.join("libonnxruntime.so.1.28.0"), true)
    };
    let runtime_guard = if bundled {
        Some(open_regular_runtime(&path)?)
    } else {
        let metadata = fs::metadata(&path).map_err(|error| {
            format!(
                "ONNX Runtime library unavailable at {}: {error}",
                path.display()
            )
        })?;
        if !metadata.file_type().is_file() {
            return Err(format!(
                "ONNX Runtime override is not a regular file: {}",
                path.display()
            ));
        }
        None
    };
    if bundled && !cache_runtime_files_present(&path) {
        return Err(format!(
            "bundled ONNX Runtime installation is incomplete in {}",
            path.parent().unwrap_or(Path::new(".")).display()
        ));
    }
    let builder = ort::init_from(&path)
        .map_err(|error| format!("cannot load ONNX Runtime from {}: {error}", path.display()))?;
    builder.commit();
    drop(runtime_guard);
    Ok(())
}

fn open_regular_runtime(path: &Path) -> Result<File, String> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| {
            format!(
                "ONNX Runtime library unavailable at {}: {error}",
                path.display()
            )
        })?;
    if !file
        .metadata()
        .map_err(|error| {
            format!(
                "cannot inspect ONNX Runtime library {}: {error}",
                path.display()
            )
        })?
        .file_type()
        .is_file()
    {
        return Err(format!(
            "ONNX Runtime cache entry is not a regular file: {}",
            path.display()
        ));
    }
    Ok(file)
}

fn cache_runtime_files_present(runtime_path: &Path) -> bool {
    is_regular_file(runtime_path).unwrap_or(false)
        && is_regular_file(&runtime_path.with_file_name("libonnxruntime_providers_shared.so"))
            .unwrap_or(false)
}

fn install_bundled_runtime(cache: &Path) -> Result<(), String> {
    let expected_runtime = include_bytes!(env!("BREADMILL_ORT_RUNTIME_LIB"));
    let expected_providers = include_bytes!(env!("BREADMILL_ORT_RUNTIME_PROVIDERS"));
    match fs::symlink_metadata(cache) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(cache) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(format!(
                        "cannot create ONNX Runtime cache {}: {error}",
                        cache.display()
                    ))
                }
            }
        }
        Err(error) => {
            return Err(format!(
                "cannot inspect ONNX Runtime cache {}: {error}",
                cache.display()
            ))
        }
    }
    let metadata = fs::symlink_metadata(cache)
        .map_err(|e| format!("cannot inspect ONNX Runtime cache {}: {e}", cache.display()))?;
    if !metadata.file_type().is_dir() {
        return Err(format!(
            "ONNX Runtime cache is not a real directory: {}",
            cache.display()
        ));
    }
    fs::set_permissions(cache, fs::Permissions::from_mode(0o700))
        .map_err(|e| format!("cannot secure ONNX Runtime cache {}: {e}", cache.display()))?;

    let _lock = RuntimeCacheLock::acquire(cache)?;
    install_library(cache, "libonnxruntime.so.1.28.0", expected_runtime)?;
    install_library(
        cache,
        "libonnxruntime_providers_shared.so",
        expected_providers,
    )?;
    Ok(())
}

fn is_regular_file(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_file()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "cannot inspect ONNX Runtime file {}: {error}",
            path.display()
        )),
    }
}

fn install_library(cache: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let installed = cache.join(name);
    match fs::symlink_metadata(&installed) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(format!(
                "ONNX Runtime cache entry is not a regular file: {}",
                installed.display()
            ));
        }
        Ok(_) => {
            if matches_embedded_file(&installed, bytes)? {
                fs::set_permissions(&installed, fs::Permissions::from_mode(0o644)).map_err(
                    |e| {
                        format!(
                            "cannot secure ONNX Runtime file {}: {e}",
                            installed.display()
                        )
                    },
                )?;
                return Ok(());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "cannot inspect ONNX Runtime file {}: {error}",
                installed.display()
            ))
        }
    }

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);
    let (temporary, mut file) = (0..100)
        .find_map(|_| {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = cache.join(format!(".{name}.{}.{}.tmp", std::process::id(), id));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
            {
                Ok(file) => Some(Ok((path, file))),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error)),
            }
        })
        .ok_or_else(|| format!("cannot allocate temporary ONNX Runtime file for {name}"))?
        .map_err(|error| {
            format!("cannot create temporary ONNX Runtime file for {name}: {error}")
        })?;

    let result = (|| {
        file.write_all(bytes)
            .map_err(|e| format!("cannot write temporary ONNX Runtime file: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("cannot sync temporary ONNX Runtime file: {e}"))?;
        file.set_permissions(fs::Permissions::from_mode(0o644))
            .map_err(|e| format!("cannot set ONNX Runtime file permissions: {e}"))?;
        fs::rename(&temporary, &installed).map_err(|e| {
            format!(
                "cannot publish ONNX Runtime file {}: {e}",
                installed.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn matches_embedded_file(path: &Path, expected: &[u8]) -> Result<bool, String> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| {
            format!(
                "cannot safely open ONNX Runtime cache entry {}: {error}",
                path.display()
            )
        })?;
    let metadata = file.metadata().map_err(|error| {
        format!(
            "cannot inspect ONNX Runtime cache entry {}: {error}",
            path.display()
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "ONNX Runtime cache entry is not a regular file: {}",
            path.display()
        ));
    }
    if metadata.len() != expected.len() as u64 {
        return Ok(false);
    }
    let mut contents = Vec::with_capacity(expected.len());
    file.take(expected.len() as u64 + 1)
        .read_to_end(&mut contents)
        .map_err(|error| {
            format!(
                "cannot read ONNX Runtime cache entry {}: {error}",
                path.display()
            )
        })?;
    Ok(contents == expected)
}

struct RuntimeCacheLock(File);

impl RuntimeCacheLock {
    fn acquire(cache: &Path) -> Result<Self, String> {
        let path = cache.join(".runtime-install.lock");
        match fs::symlink_metadata(&path) {
            Ok(metadata) if !metadata.file_type().is_file() => {
                return Err(format!(
                    "ONNX Runtime install lock is not a regular file: {}",
                    path.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "cannot inspect ONNX Runtime install lock {}: {error}",
                    path.display()
                ));
            }
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&path)
            .map_err(|e| {
                format!(
                    "cannot open ONNX Runtime install lock {}: {e}",
                    path.display()
                )
            })?;
        let metadata = file
            .metadata()
            .map_err(|e| format!("cannot inspect ONNX Runtime install lock: {e}"))?;
        if !metadata.file_type().is_file() {
            return Err(format!(
                "ONNX Runtime install lock is not a regular file: {}",
                path.display()
            ));
        }
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|e| format!("cannot secure ONNX Runtime install lock: {e}"))?;
        // SAFETY: flock only operates on this open file descriptor. The file is
        // retained by RuntimeCacheLock until installation and verification end.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if result != 0 {
            return Err(format!(
                "cannot lock ONNX Runtime cache: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self(file))
    }
}

impl Drop for RuntimeCacheLock {
    fn drop(&mut self) {
        // SAFETY: this descriptor remains open for the lifetime of self.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

// ---- Backend -> bread_onnx::Provider ----------------------------------------
//
// `bread_onnx::session::build_session` (via `EmbeddingSession::load`) is the
// shared session-builder + EP-fallback + loud-logging code every EP branch
// below used to hand-roll separately (`configure_eps`/`npu_session`/
// `rocm_session`/`cuda_session`/`openvino_session`). What's genuinely
// specific to this crate: its cargo feature gates (npu/rocm/cuda/openvino),
// and NPU's `vaip_config.json` discovery, stays here.

fn to_provider(backend: Backend) -> Result<Provider, String> {
    match backend {
        Backend::Cpu => Ok(Provider::Cpu),
        Backend::Npu { cache_dir } => npu_provider(cache_dir),
        Backend::Rocm => rocm_provider(),
        Backend::Cuda => cuda_provider(),
        Backend::OpenVino { cache_dir } => Ok(Provider::OpenVino {
            device_type: "GPU".to_string(),
            cache_dir,
        }),
    }
}

#[cfg(feature = "npu")]
fn npu_provider(cache_dir: PathBuf) -> Result<Provider, String> {
    let vaip_config = find_vaip_config()?;
    if std::env::var("ORT_DYLIB_PATH").is_err() {
        eprintln!(
            "breadmill: hint: set ORT_DYLIB_PATH to the Ryzen AI SDK ORT, e.g.:\n  \
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

#[cfg(test)]
mod runtime_cache_tests {
    use super::*;
    use std::{
        os::unix::fs::FileTypeExt,
        sync::{Arc, Barrier},
        thread,
    };

    #[test]
    fn fifo_cache_entry_is_rejected_without_opening_it() {
        let temp =
            std::env::temp_dir().join(format!("breadmill-runtime-fifo-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        let fifo = temp.join("runtime.so");
        assert_eq!(
            unsafe {
                libc::mkfifo(
                    std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes())
                        .unwrap()
                        .as_ptr(),
                    0o600,
                )
            },
            0
        );
        assert!(fs::symlink_metadata(&fifo).unwrap().file_type().is_fifo());
        assert!(!is_regular_file(&fifo).unwrap());
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn concurrent_publication_leaves_complete_libraries_with_private_directory() {
        let temp = std::env::temp_dir().join(format!(
            "breadmill-runtime-concurrent-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let cache = temp.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    install_bundled_runtime(&cache).unwrap();
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let runtime = temp.join("libonnxruntime.so.1.28.0");
        let providers = temp.join("libonnxruntime_providers_shared.so");
        assert_eq!(
            fs::read(runtime).unwrap(),
            include_bytes!(env!("BREADMILL_ORT_RUNTIME_LIB"))
        );
        assert_eq!(
            fs::read(providers).unwrap(),
            include_bytes!(env!("BREADMILL_ORT_RUNTIME_PROVIDERS"))
        );
        assert_eq!(
            fs::metadata(&temp).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(temp.join("libonnxruntime.so.1.28.0"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o644
        );
        assert_eq!(
            fs::metadata(temp.join("libonnxruntime_providers_shared.so"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o644
        );
        fs::remove_dir_all(temp).unwrap();
    }
}
