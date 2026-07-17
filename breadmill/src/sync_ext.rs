//! Poison-tolerant `Mutex` locking.
//!
//! A panic on any thread while holding `SharedState::store` or
//! `SharedState::embedder` (e.g. an untrapped ONNX Runtime panic — only PDF
//! extraction is `catch_unwind`-guarded, see `extract.rs`) poisons the
//! `Mutex`. Every subsequent plain `.lock().unwrap()` — in the indexer *and*
//! in every query handler in `serve.rs` — would then immediately panic too,
//! silently bricking indexing and search until the daemon is restarted by
//! hand.
//!
//! `Mutex` poisoning exists to flag "the data guarded by this lock might be
//! in an inconsistent state," but every lock scope in this crate is a short,
//! single-step SQLite call or usearch operation — nothing here spans a
//! multi-step invariant across a single `lock()` call — so recovering the
//! guard and logging loudly is a reasonable trade here: continuing to serve
//! (and re-attempting the operation that panicked, on the next file/query)
//! beats a daemon that silently stops answering everything after one panic.

use std::sync::{Mutex, MutexGuard};

pub trait MutexExt<T> {
    /// Like `.lock().unwrap()`, but recovers from a poisoned mutex instead
    /// of panicking again — logs once per recovery so it's visible in the
    /// daemon's own output, not just silently swallowed.
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        match self.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                eprintln!(
                    "breadmill: WARNING: a mutex was poisoned by a panic on another thread; \
                     recovering it and continuing instead of cascading the panic"
                );
                poisoned.into_inner()
            }
        }
    }
}
