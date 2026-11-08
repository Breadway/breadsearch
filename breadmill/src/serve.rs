use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    sync::{atomic::Ordering, Arc},
};

use breadsearch_shared::{Request, Response, StatusInfo};

use crate::indexer::SharedState;
use crate::sync_ext::MutexExt;

pub fn run(socket_path: &Path, state: Arc<SharedState>, snippet_len: usize, search_limit: usize) {
    let _ = std::fs::remove_file(socket_path);

    let listener = match UnixListener::bind(socket_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("breadmill: bind {}: {}", socket_path.display(), e);
            return;
        }
    };

    eprintln!("breadmill: listening on {}", socket_path.display());

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    handle(s, state, snippet_len, search_limit);
                });
            }
            Err(e) => eprintln!("breadmill: accept error: {}", e),
        }
    }
}

fn handle(stream: UnixStream, state: Arc<SharedState>, snippet_len: usize, search_limit: usize) {
    let stream_write = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };

    let mut reader = BufReader::new(&stream);
    let mut writer = std::io::BufWriter::new(stream_write);

    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }

    let response = match serde_json::from_str::<Request>(line.trim()) {
        Ok(req) => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dispatch(req, &state, snippet_len, search_limit)
            }));
            match r {
                Ok(resp) => resp,
                Err(_) => Response::Error {
                    message: "internal error".into(),
                },
            }
        }
        Err(e) => Response::Error {
            message: e.to_string(),
        },
    };

    if let Ok(mut json) = serde_json::to_string(&response) {
        json.push('\n');
        let _ = writer.write_all(json.as_bytes());
        let _ = writer.flush();
    }
}

/// Count a query as received at dispatch entry.
///
/// The counter saturates instead of wrapping, so a daemon that has answered
/// `u64::MAX` queries keeps reporting a huge number rather than looking idle.
/// Counting belongs here, before any model check, so a query that finds the
/// model unloaded and asks for a reload is still visible to status pollers.
fn record_query_received(state: &SharedState) {
    let _ = state
        .queries_served
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |served| {
            Some(served.saturating_add(1))
        });
}

fn dispatch(
    req: Request,
    state: &std::sync::Arc<SharedState>,
    snippet_len: usize,
    search_limit: usize,
) -> Response {
    match req {
        Request::Query { query, limit } => dispatch_query(
            state,
            &query,
            limit,
            snippet_len,
            search_limit,
            // Probe the model only after the query has been counted, so every
            // arriving query is visible even when the probe sends it into the
            // demand-reload path.
            || state.embedder.lock_recover().is_some(),
            |query| {
                let mut embedder = state.embedder.lock_recover();
                match embedder.as_mut() {
                    Some(embedder) => embedder.embed_query(query),
                    None => Err("embedder unavailable".into()),
                }
            },
        ),

        Request::Status => Response::StatusInfo(StatusInfo {
            indexed: state.indexed.load(Ordering::Relaxed),
            pending: state.pending.load(Ordering::Relaxed),
            model_ready: state.model_ready.load(Ordering::Relaxed),
            in_flight: state.in_flight.load(Ordering::Relaxed),
            loading: state.loading.load(Ordering::Relaxed),
            model_bytes: match state.model_bytes.load(Ordering::Relaxed) {
                0 => None,
                bytes => Some(bytes),
            },
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            queries_served: Some(state.queries_served.load(Ordering::Relaxed)),
            model_error: state.model_error.lock_recover().clone(),
        }),

        Request::Reindex => {
            state.reindex_signal.store(true, Ordering::Relaxed);
            Response::Ok
        }

        Request::UnloadModel => match crate::model::unload(state) {
            Ok(()) => Response::Ok,
            Err(reason) => Response::Error {
                message: format!("busy: {reason}"),
            },
        },

        Request::LoadModel => {
            let params = state.loader.lock_recover().clone();
            match params {
                Some(params) => {
                    crate::model::load_in_background(state, &params);
                    // Accepted, not completed: the load runs in the
                    // background and readiness is polled through Status.
                    Response::Ok
                }
                None => Response::Error {
                    message: "model loader not configured".into(),
                },
            }
        }
    }
}

fn dispatch_query<M, F>(
    state: &std::sync::Arc<SharedState>,
    query: &str,
    limit: usize,
    snippet_len: usize,
    search_limit: usize,
    model_present: M,
    embed: F,
) -> Response
where
    M: FnOnce() -> bool,
    F: FnOnce(&str) -> Result<Vec<f32>, String>,
{
    // Count at dispatch entry, before the model-present branch. A query that
    // arrives while the model is unloaded still arrived, and its demand
    // reload should not be the only trace a poll might miss.
    record_query_received(state);

    // Demand reload: a search that finds the model gone starts it loading
    // again and says the model is loading instead of failing silently. The
    // search itself does not wait: it reports busy, and the next one finds a
    // warm model. Missing model files are not a reload case: they were never
    // fetched.
    if !model_present() {
        let params = state.loader.lock_recover().clone();
        return match params {
            Some(params) if params.model_path.exists() => {
                crate::model::load_in_background(state, &params);
                Response::Error {
                    message: "model loading, retry shortly".into(),
                }
            }
            _ => Response::Error {
                message: "model not ready, run breadmill --fetch-model".into(),
            },
        };
    }

    state.in_flight.fetch_add(1, Ordering::Relaxed);
    let embedding = match embed(query) {
        Ok(embedding) => embedding,
        Err(e) => {
            state.in_flight.fetch_sub(1, Ordering::Relaxed);
            return Response::Error { message: e };
        }
    };
    state.in_flight.fetch_sub(1, Ordering::Relaxed);

    let limit = limit.min(search_limit).max(1);
    let store = state.store.lock_recover();

    match store.search(&embedding, limit, snippet_len) {
        Ok(hits) => Response::Hits { hits },
        Err(e) => Response::Error { message: e },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::Backend;
    use crate::model::ModelParams;
    use crate::store::Store;
    use std::path::PathBuf;

    fn test_state(name: &str, dim: usize) -> (Arc<SharedState>, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "breadmill-serve-test-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let store = Store::open(&dir, dim).unwrap();
        (Arc::new(SharedState::new(store)), dir)
    }

    #[test]
    fn served_query_counts_once() {
        let (state, dir) = test_state("served", 4);

        let response = dispatch_query(&state, "hello", 5, 80, 10, || true, |_| Ok(vec![0.0; 4]));

        assert!(matches!(response, Response::Hits { .. }));
        assert_eq!(state.queries_served.load(Ordering::Relaxed), 1);
        assert_eq!(state.in_flight.load(Ordering::Relaxed), 0);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn demand_reload_query_counts_once() {
        let (state, dir) = test_state("reload", 4);
        let model_path = dir.join("model.onnx");
        std::fs::write(&model_path, b"not a model").unwrap();
        *state.loader.lock_recover() = Some(ModelParams {
            model_path,
            tokenizer_path: dir.join("tokenizer.json"),
            dim: 4,
            backend: Backend::Cpu,
        });
        // Keep the loader from spawning a real native load: the counter's
        // placement before this branch is what the test checks, and an
        // already-running load takes the same demand-reload branch.
        state.loading.store(true, Ordering::Relaxed);

        let response = dispatch_query(
            &state,
            "hello",
            5,
            80,
            10,
            || false,
            |_| -> Result<Vec<f32>, String> {
                panic!("a query that triggers a reload must not embed")
            },
        );

        match response {
            Response::Error { message } => assert!(message.contains("model loading")),
            other => panic!("unexpected response: {other:?}"),
        }
        assert_eq!(state.queries_served.load(Ordering::Relaxed), 1);
        assert_eq!(state.in_flight.load(Ordering::Relaxed), 0);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn query_counter_saturates() {
        let (state, dir) = test_state("saturate", 4);
        state.queries_served.store(u64::MAX, Ordering::Relaxed);

        record_query_received(&state);
        record_query_received(&state);

        assert_eq!(state.queries_served.load(Ordering::Relaxed), u64::MAX);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn status_reports_queries_without_counting_itself() {
        let (state, dir) = test_state("status", 4);
        state.queries_served.store(7, Ordering::Relaxed);

        let response = dispatch(Request::Status, &state, 80, 10);

        match response {
            Response::StatusInfo(status) => assert_eq!(status.queries_served, Some(7)),
            other => panic!("unexpected response: {other:?}"),
        }
        assert_eq!(state.queries_served.load(Ordering::Relaxed), 7);

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_request_does_not_count() {
        let (state, dir) = test_state("invalid", 4);
        let (client, server) = UnixStream::pair().unwrap();
        let handle_state = Arc::clone(&state);
        let thread = std::thread::spawn(move || handle(server, handle_state, 80, 10));

        let mut writer = client.try_clone().unwrap();
        writer.write_all(b"not a request\n").unwrap();
        writer.flush().unwrap();

        let mut reader = BufReader::new(&client);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let response: Response = serde_json::from_str(line.trim()).unwrap();
        assert!(matches!(response, Response::Error { .. }));
        thread.join().unwrap();

        assert_eq!(state.queries_served.load(Ordering::Relaxed), 0);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
