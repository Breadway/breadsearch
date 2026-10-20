//! Model lifecycle: load in the background, unload on request.
//!
//! The embedder lives behind the shared state's mutex as an `Option`, so
//! dropping the session is always safe: a query that arrives mid-unload either
//! waited on the mutex and sees the model gone (an honest error, never a torn
//! read), or finished first. Unload is refused while queries are in flight so
//! no running search is ruined to free the memory; the indexer needs no such
//! refusal because it already stops cleanly when the model disappears
//! mid-scan, keeping every chunk it stored. A query that arrives while the
//! model is gone starts a background reload and says the model is loading, so
//! the search that triggered the reload knows to retry and the next one finds
//! a warm model.

use std::{
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};

use crate::embed::{Backend, OrtEmbedder};
use crate::indexer::SharedState;
use crate::sync_ext::MutexExt;

/// Everything a background load needs, captured once at startup.
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// The ONNX model file.
    pub model_path: PathBuf,
    /// Its tokenizer.
    pub tokenizer_path: PathBuf,
    /// The embedding dimension from configuration.
    pub dim: usize,
    /// The compute backend from configuration or CLI flags.
    pub backend: Backend,
}

/// The model file size in bytes, or `None` when it cannot be read.
///
/// The file is memory-mapped when loaded, so this is an estimate of model
/// bytes and never a measurement of host RAM use. It is reported so a memory
/// manager can size the relief a release is expected to free.
pub fn model_file_bytes(model_path: &Path) -> Option<u64> {
    std::fs::metadata(model_path).ok().map(|meta| meta.len())
}

/// Start loading the model in the background, unless it is already loaded or
/// loading.
///
/// Idempotent: a second call while a load is in flight does nothing. Success
/// stores the embedder and marks the model ready; failure leaves the model
/// absent (and any previous `model_ready` untouched) so a later query or an
/// explicit load can try again.
pub fn load_in_background(state: &Arc<SharedState>, params: &ModelParams) {
    if state.model_ready.load(Ordering::Relaxed) {
        return;
    }
    if state.loading.swap(true, Ordering::Relaxed) {
        return;
    }
    *state.model_error.lock_recover() = None;
    let state = Arc::clone(state);
    let params = params.clone();
    std::thread::spawn(move || {
        eprintln!("breadmill: loading model...");
        match OrtEmbedder::load(
            &params.model_path,
            &params.tokenizer_path,
            params.dim,
            params.backend,
        ) {
            Ok(embedder) => {
                let bytes = model_file_bytes(&params.model_path);
                *state.embedder.lock_recover() = Some(embedder);
                if let Some(bytes) = bytes {
                    state.model_bytes.store(bytes, Ordering::Relaxed);
                }
                state.model_ready.store(true, Ordering::Relaxed);
                state.loading.store(false, Ordering::Relaxed);
                eprintln!("breadmill: model loaded");
            }
            Err(e) => {
                *state.model_error.lock_recover() = Some(e.clone());
                state.loading.store(false, Ordering::Relaxed);
                eprintln!("breadmill: model load failed: {e}");
            }
        }
    });
}

/// Drop the loaded session, freeing its backend allocations.
///
/// Refused while queries are in flight: those searches hold no lock across
/// the whole request, so yanking the model under them would turn running
/// work into errors. The indexer is not counted because it stops cleanly at
/// the next chunk when the model disappears. On success the model is gone
/// and `model_ready` is false; user configuration, model files, and every
/// indexed chunk are untouched.
pub fn unload(state: &SharedState) -> Result<(), String> {
    // A load in flight would resurrect the model the moment it finishes, so
    // unloading under it is refused rather than raced.
    if state.loading.load(Ordering::Relaxed) {
        return Err("model loading".to_string());
    }
    let in_flight = state.in_flight.load(Ordering::Relaxed);
    if in_flight > 0 {
        return Err(format!("{in_flight} queries in flight"));
    }
    *state.embedder.lock_recover() = None;
    state.model_ready.store(false, Ordering::Relaxed);
    eprintln!("breadmill: model unloaded, backend allocations released");
    Ok(())
}
