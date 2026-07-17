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
                Err(_) => Response::Error { message: "internal error".into() },
            }
        }
        Err(e) => Response::Error { message: e.to_string() },
    };

    if let Ok(mut json) = serde_json::to_string(&response) {
        json.push('\n');
        let _ = writer.write_all(json.as_bytes());
        let _ = writer.flush();
    }
}

fn dispatch(
    req: Request,
    state: &SharedState,
    snippet_len: usize,
    search_limit: usize,
) -> Response {
    use std::sync::atomic::Ordering;

    match req {
        Request::Query { query, limit } => {
            if !state.model_ready.load(Ordering::Relaxed) {
                return Response::Error {
                    message: "model not ready — run breadmill --fetch-model".into(),
                };
            }

            let embedding = {
                let mut embedder = state.embedder.lock_recover();
                match embedder.as_mut() {
                    Some(e) => match e.embed_query(&query) {
                        Ok(v) => v,
                        Err(e) => return Response::Error { message: e },
                    },
                    None => {
                        return Response::Error {
                            message: "embedder unavailable".into(),
                        }
                    }
                }
            };

            let limit = limit.min(search_limit).max(1);
            let store = state.store.lock_recover();

            match store.search(&embedding, limit, snippet_len) {
                Ok(hits) => Response::Hits { hits },
                Err(e) => Response::Error { message: e },
            }
        }

        Request::Status => {
            use std::sync::atomic::Ordering;
            Response::StatusInfo(StatusInfo {
                indexed: state.indexed.load(Ordering::Relaxed),
                pending: state.pending.load(Ordering::Relaxed),
                model_ready: state.model_ready.load(Ordering::Relaxed),
            })
        }

        Request::Reindex => {
            use std::sync::atomic::Ordering;
            state.reindex_signal.store(true, Ordering::Relaxed);
            Response::Ok
        }
    }
}
