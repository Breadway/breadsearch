use std::path::Path;

use rusqlite::{Connection, params};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

pub struct Store {
    pub conn: Connection,
    pub index: Index,
    pub dim: usize,
}

// usearch::Index wraps a raw C++ pointer; access is serialized by the Mutex<Store>.
unsafe impl Send for Store {}

#[derive(Debug)]
pub struct FileMeta {
    pub path: String,
    pub mtime: i64,
    pub hash: String,
}

impl Store {
    pub fn open(state_dir: &Path, dim: usize) -> Result<Self, String> {
        std::fs::create_dir_all(state_dir).map_err(|e| e.to_string())?;

        let db_path = state_dir.join("meta.db");
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS files (
                 path  TEXT PRIMARY KEY,
                 mtime INTEGER NOT NULL,
                 hash  TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS chunks (
                 id          INTEGER PRIMARY KEY,
                 path        TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                 chunk_text  TEXT NOT NULL,
                 chunk_start INTEGER NOT NULL,
                 chunk_end   INTEGER NOT NULL
             );",
        )
        .map_err(|e| e.to_string())?;

        let idx_path = state_dir.join("vectors.usearch");
        let options = IndexOptions {
            dimensions: dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
            multi: false,
        };
        let index = new_index(&options).map_err(|e| e.to_string())?;

        if idx_path.exists() {
            index
                .load(idx_path.to_str().unwrap())
                .map_err(|e| e.to_string())?;
        } else {
            index.reserve(4096).map_err(|e| e.to_string())?;
        }

        Ok(Self { conn, index, dim })
    }

    // ---- file state ---------------------------------------------------------

    pub fn all_files(&self) -> Result<Vec<FileMeta>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, mtime, hash FROM files")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |row| {
                Ok(FileMeta {
                    path: row.get(0)?,
                    mtime: row.get(1)?,
                    hash: row.get(2)?,
                })
            })
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn upsert_file(&self, path: &str, mtime: i64, hash: &str) -> Result<(), String> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO files (path, mtime, hash) VALUES (?1, ?2, ?3)",
                params![path, mtime, hash],
            )
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Delete a file and all its chunks from SQLite; also remove chunk vectors.
    pub fn delete_file(&mut self, path: &str) -> Result<(), String> {
        // Collect chunk IDs before deletion for usearch removal
        let ids = self.chunk_ids_for(path)?;

        self.conn
            .execute("DELETE FROM files WHERE path = ?1", params![path])
            .map_err(|e| e.to_string())?;

        for id in ids {
            let _ = self.index.remove(id); // best-effort; stale entries are harmless
        }

        Ok(())
    }

    fn chunk_ids_for(&self, path: &str) -> Result<Vec<u64>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM chunks WHERE path = ?1")
            .map_err(|e| e.to_string())?;
        let ids = stmt
            .query_map(params![path], |row| row.get::<_, i64>(0))
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .map(|id| id as u64)
            .collect();
        Ok(ids)
    }

    // ---- chunk operations ---------------------------------------------------

    pub fn insert_chunk(
        &mut self,
        path: &str,
        text: &str,
        start: usize,
        end: usize,
        embedding: &[f32],
    ) -> Result<u64, String> {
        self.conn
            .execute(
                "INSERT INTO chunks (path, chunk_text, chunk_start, chunk_end)
                 VALUES (?1, ?2, ?3, ?4)",
                params![path, text, start as i64, end as i64],
            )
            .map_err(|e| e.to_string())?;

        let id = self.conn.last_insert_rowid() as u64;

        // Grow index if needed
        if self.index.size() + 1 > self.index.capacity() {
            self.index
                .reserve(self.index.capacity() + 4096)
                .map_err(|e| e.to_string())?;
        }

        self.index.add(id, embedding).map_err(|e| e.to_string())?;

        Ok(id)
    }

    pub fn chunk_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_or(0) as usize
    }

    // ---- query --------------------------------------------------------------

    pub fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        snippet_len: usize,
    ) -> Result<Vec<breadsearch_shared::Hit>, String> {
        let results = self
            .index
            .search(embedding, limit)
            .map_err(|e| e.to_string())?;

        let mut hits = Vec::new();

        for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
            // Convert cosine distance → similarity score (higher = better)
            let score = 1.0 - distance;

            let maybe_chunk = self
                .conn
                .query_row(
                    "SELECT chunk_text, path FROM chunks WHERE id = ?1",
                    params![*key as i64],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .ok();

            if let Some((chunk_text, path)) = maybe_chunk {
                let snippet = truncate_to_chars(&chunk_text, snippet_len);
                let title = std::path::Path::new(&path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&path)
                    .to_string();

                hits.push(breadsearch_shared::Hit {
                    title,
                    path,
                    snippet,
                    score,
                });
            }
        }

        Ok(hits)
    }

    // ---- persistence --------------------------------------------------------

    pub fn save_index(&self, state_dir: &Path) -> Result<(), String> {
        let idx_path = state_dir.join("vectors.usearch");
        self.index
            .save(idx_path.to_str().unwrap())
            .map_err(|e| e.to_string())
    }
}

fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{}…", truncated.trim_end())
}
