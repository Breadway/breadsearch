use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}},
    time::{Duration, Instant, UNIX_EPOCH},
};

use ignore::WalkBuilder;
use notify::{RecommendedWatcher, RecursiveMode, Watcher, EventKind};
use sha2::{Digest, Sha256};

use crate::{embed::OrtEmbedder, extract, chunk, power, store::Store, sync_ext::MutexExt};

pub struct SharedState {
    pub store: Mutex<Store>,
    pub embedder: Mutex<Option<OrtEmbedder>>,
    pub model_ready: AtomicBool,
    pub indexed: AtomicUsize,
    pub pending: AtomicUsize,
    pub reindex_signal: AtomicBool,
}

impl SharedState {
    pub fn new(store: Store) -> Self {
        SharedState {
            store: Mutex::new(store),
            embedder: Mutex::new(None),
            model_ready: AtomicBool::new(false),
            indexed: AtomicUsize::new(0),
            pending: AtomicUsize::new(0),
            reindex_signal: AtomicBool::new(false),
        }
    }
}

pub struct Indexer {
    state: Arc<SharedState>,
    config: breadsearch_shared::Config,
    state_dir: PathBuf,
}

impl Indexer {
    pub fn new(state: Arc<SharedState>, config: breadsearch_shared::Config, state_dir: PathBuf) -> Self {
        Indexer { state, config, state_dir }
    }

    pub fn run(self) {
        self.initial_scan();
        self.watch_loop();
    }

    /// True when embedding should be skipped right now: the user turned
    /// indexing off entirely, or the machine is on battery and
    /// `power.run_on_battery` is not set. Cheap sysfs reads — safe to call
    /// per-file and on every watch_loop tick.
    fn indexing_paused(&self) -> bool {
        if !self.config.power.enabled {
            return true;
        }
        !self.config.power.run_on_battery && !power::on_ac_power()
    }

    pub fn full_reindex(&self) {
        eprintln!("breadmill: full reindex triggered");
        {
            let mut store = self.state.store.lock_recover();
            // Clear all state
            let _ = store.conn.execute_batch("DELETE FROM chunks; DELETE FROM files;");
            let _ = store.index.reserve(4096);
        }
        self.initial_scan();
    }

    fn initial_scan(&self) {
        eprintln!("breadmill: scanning roots...");

        let roots: Vec<PathBuf> = self.config.index.roots
            .iter()
            .map(|r| expand_home(r))
            .collect();

        let excludes: Vec<PathBuf> = self.config.index.excludes
            .iter()
            .map(|r| expand_home(r))
            .collect();

        // Snapshot existing indexed files
        let known: HashMap<String, (i64, String)> = {
            let store = self.state.store.lock_recover();
            store.all_files()
                .unwrap_or_default()
                .into_iter()
                .map(|f| (f.path, (f.mtime, f.hash)))
                .collect()
        };

        let mut seen: HashSet<String> = HashSet::new();
        let max_bytes = (self.config.index.max_file_mb * 1024.0 * 1024.0) as u64;

        for root in &roots {
            if !root.exists() {
                continue;
            }

            for entry in WalkBuilder::new(root)
                .hidden(false)
                .ignore(true)
                .git_ignore(true)
                .build()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
            {
                let path = entry.path();

                if excludes.iter().any(|excl| path.starts_with(excl)) {
                    continue;
                }

                if !self.is_indexed_extension(path) {
                    continue;
                }

                let meta = match fs::metadata(path) {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                if meta.len() > max_bytes {
                    continue;
                }

                let path_str = path.to_string_lossy().into_owned();
                seen.insert(path_str.clone());

                let mtime = mtime_secs(&meta);

                if let Some((known_mtime, _)) = known.get(&path_str) {
                    if *known_mtime == mtime {
                        continue; // unchanged
                    }
                }

                self.state.pending.fetch_add(1, Ordering::Relaxed);
                self.index_file(path, &path_str, mtime);
                self.state.pending.fetch_sub(1, Ordering::Relaxed);
            }
        }

        // Drop files that were deleted
        let to_delete: Vec<String> = known
            .keys()
            .filter(|p| !seen.contains(*p))
            .cloned()
            .collect();

        if !to_delete.is_empty() {
            let mut store = self.state.store.lock_recover();
            for path in to_delete {
                eprintln!("breadmill: removing deleted file: {}", path);
                let _ = store.delete_file(&path);
            }
        }

        let count = {
            let store = self.state.store.lock_recover();
            let n = store.chunk_count();
            let _ = store.save_index(&self.state_dir);
            n
        };
        self.state.indexed.store(count, Ordering::Relaxed);
        eprintln!("breadmill: initial scan done — {} chunks indexed", count);
    }

    fn watch_loop(self) {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut watcher: RecommendedWatcher = match notify::recommended_watcher(tx) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("breadmill: watcher init failed: {}", e);
                return;
            }
        };

        for root in self.config.index.roots.iter().map(|r| expand_home(r)) {
            if root.exists() {
                let _ = watcher.watch(&root, RecursiveMode::Recursive);
            }
        }

        let mut pending_paths: HashSet<PathBuf> = HashSet::new();
        let mut last_event = Instant::now();
        let quiet = Duration::from_secs(2);
        let mut was_paused = self.indexing_paused();
        let mut last_power_check = Instant::now();

        eprintln!("breadmill: watching for changes");

        loop {
            // Drain the reindex signal
            if self.state.reindex_signal.swap(false, Ordering::Relaxed) {
                self.full_reindex();
            }

            // Re-check the power gate periodically (sysfs reads are cheap but
            // no need to do it every 500ms tick). On a paused->active
            // transition, re-run the incremental scan to catch up anything
            // skipped while gated.
            if last_power_check.elapsed() >= Duration::from_secs(30) {
                last_power_check = Instant::now();
                let now_paused = self.indexing_paused();
                if was_paused && !now_paused {
                    eprintln!("breadmill: power gate opened — resuming indexing");
                    self.initial_scan();
                }
                was_paused = now_paused;
            }

            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok(Ok(event)) => {
                    match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                            for p in event.paths {
                                pending_paths.insert(p);
                            }
                            last_event = Instant::now();
                        }
                        _ => {}
                    }
                }
                Ok(Err(e)) => eprintln!("breadmill: watch error: {}", e),
                Err(_) => {} // timeout — check quiet period
            }

            if !pending_paths.is_empty() && last_event.elapsed() >= quiet {
                for path in pending_paths.drain() {
                    self.handle_fs_event(&path);
                }
                let count = {
                    let store = self.state.store.lock_recover();
                    let n = store.chunk_count();
                    let _ = store.save_index(&self.state_dir);
                    n
                };
                self.state.indexed.store(count, Ordering::Relaxed);
            }
        }
    }

    fn handle_fs_event(&self, path: &Path) {
        let excludes: Vec<PathBuf> = self.config.index.excludes
            .iter()
            .map(|r| expand_home(r))
            .collect();

        if excludes.iter().any(|excl| path.starts_with(excl)) {
            return;
        }

        if !path.is_file() {
            let path_str = path.to_string_lossy().into_owned();
            // File deleted — remove from index
            let mut store = self.state.store.lock_recover();
            let _ = store.delete_file(&path_str);
            return;
        }

        if !self.is_indexed_extension(path) {
            return;
        }

        let meta = match fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return,
        };

        let max_bytes = (self.config.index.max_file_mb * 1024.0 * 1024.0) as u64;
        if meta.len() > max_bytes {
            return;
        }

        let path_str = path.to_string_lossy().into_owned();
        let mtime = mtime_secs(&meta);

        self.state.pending.fetch_add(1, Ordering::Relaxed);
        self.index_file(path, &path_str, mtime);
        self.state.pending.fetch_sub(1, Ordering::Relaxed);
    }

    fn index_file(&self, path: &Path, path_str: &str, mtime: i64) {
        if self.indexing_paused() {
            // Leave the file unrecorded so it's picked up again once indexing
            // resumes (initial_scan/watch_loop treat it as not-yet-indexed).
            return;
        }

        eprintln!("breadmill: extracting {}", path_str);
        let text = match extract::extract(path) {
            Ok(t) if !t.trim().is_empty() => t,
            Ok(_) => return,
            Err(e) => {
                eprintln!("breadmill: extract {}: {}", path_str, e);
                return;
            }
        };

        let hash = sha256_str(text.as_bytes());

        // Check if hash changed (catches content changes without mtime change)
        {
            let store = self.state.store.lock_recover();
            if let Ok(files) = store.all_files() {
                if files.iter().any(|f| f.path == path_str && f.hash == hash) {
                    return;
                }
            }
        }

        // 2000 char cap keeps even minified single-line files to ~500–2000 tokens,
        // avoiding quadratic attention blowup while still splitting at word boundaries
        // for natural-language files.
        let chunks = chunk::chunk_text(&text, 400, 80, 2_000);
        eprintln!("breadmill: embedding {} ({} chars, {} chunks)", path_str, text.len(), chunks.len());

        if !self.state.model_ready.load(Ordering::Relaxed) {
            eprintln!("breadmill: model not ready, skipping embed for {}", path_str);
            return;
        }

        // Confirm the embedder is actually present before committing to
        // clearing this file's old chunks below — same check as before,
        // just without holding the embedder lock past this one glance (see
        // the per-chunk locking in the loop for why).
        if self.state.embedder.lock_recover().is_none() {
            return;
        }

        {
            let mut store = self.state.store.lock_recover();
            let _ = store.delete_file(path_str); // remove old chunks/vectors first
        }

        let mut any_ok = false;
        let mut chunks_added = 0usize;

        for (i, chunk) in chunks.iter().enumerate() {
            eprintln!("breadmill: embed chunk {}/{} ({} chars) for {}", i + 1, chunks.len(), chunk.text.len(), path_str);
            // Lock the embedder only around this single chunk's embed call —
            // this used to be held for the whole file's chunk loop, so one
            // large file needing a fresh MIGraphX JIT compile (60-120s for a
            // new sequence length) could block every query (serve.rs locks
            // this same mutex) for the entire file, not just one chunk.
            let embed_result = match self.state.embedder.lock_recover().as_mut() {
                Some(embedder) => embedder.embed_document(&chunk.text),
                None => break, // model was unloaded mid-scan; stop here
            };
            match embed_result {
                Ok(embedding) => {
                    let mut store = self.state.store.lock_recover();
                    // Ensure file row exists before inserting chunks (FK constraint)
                    let _ = store.upsert_file(path_str, mtime, &hash);
                    let _ = store.insert_chunk(
                        path_str,
                        &chunk.text,
                        chunk.start,
                        chunk.end,
                        &embedding,
                    );
                    any_ok = true;
                    chunks_added += 1;
                }
                Err(e) => eprintln!("breadmill: embed error for {}: {}", path_str, e),
            }
        }

        if !any_ok {
            eprintln!("breadmill: no chunks embedded for {}", path_str);
            // Record the file so the mtime+hash check skips it on the next startup
            // rather than re-entering the same embed-fail loop.
            let store = self.state.store.lock_recover();
            let _ = store.upsert_file(path_str, mtime, &hash);
        } else {
            // Increment live so `status` reflects progress before the full scan ends.
            self.state.indexed.fetch_add(chunks_added, Ordering::Relaxed);
            eprintln!("breadmill: done {} ({} chunks indexed)", path_str, chunks_added);
        }
    }

    fn is_indexed_extension(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| {
                self.config
                    .index
                    .extensions
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(ext))
            })
            .unwrap_or(false)
    }
}

fn mtime_secs(meta: &fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn sha256_str(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

pub fn expand_home(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(&path[2..])
    } else {
        PathBuf::from(path)
    }
}
