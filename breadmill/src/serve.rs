use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    sync::Arc,
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

fn dispatch(
    req: Request,
    state: &std::sync::Arc<SharedState>,
    snippet_len: usize,
    search_limit: usize,
) -> Response {
    use std::sync::atomic::Ordering;

    match req {
        Request::Query { query, limit } => {
            // Demand reload: a search that finds the model gone starts it
            // loading again and says the model is loading instead of failing
            // silently. The search itself does not wait: it reports busy, and
            // the next one finds a warm model. Missing model files are not a
            // reload case: they were never fetched.
            if state.embedder.lock_recover().is_none() {
                let params = state.loader.lock_recover().clone();
                match params {
                    Some(params) if params.model_path.exists() => {
                        crate::model::load_in_background(state, &params);
                        return Response::Error {
                            message: "model loading, retry shortly".into(),
                        };
                    }
                    _ => {
                        return Response::Error {
                            message: "model not ready, run breadmill --fetch-model".into(),
                        };
                    }
                }
            }

            state.in_flight.fetch_add(1, Ordering::Relaxed);
            state.queries_served.fetch_add(1, Ordering::Relaxed);
            let embedding = {
                let mut embedder = state.embedder.lock_recover();
                match embedder.as_mut() {
                    Some(e) => match e.embed_query(&query) {
                        Ok(v) => v,
                        Err(e) => {
                            state.in_flight.fetch_sub(1, Ordering::Relaxed);
                            return Response::Error { message: e };
                        }
                    },
                    None => {
                        state.in_flight.fetch_sub(1, Ordering::Relaxed);
                        return Response::Error {
                            message: "embedder unavailable".into(),
                        };
                    }
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
