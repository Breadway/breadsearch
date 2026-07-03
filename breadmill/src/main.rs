use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

use breadsearch_shared::{Request, Response};

mod chunk;
mod embed;
mod extract;
mod indexer;
mod power;
mod serve;
mod store;

use embed::{Backend, OrtEmbedder};
use indexer::{Indexer, SharedState};
use store::Store;

const MODEL_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json";

fn main() {
    // Surfaces ort's EP-registration warnings/errors (e.g. a GPU EP silently
    // falling back to CPU) by default, without requiring RUST_LOG to be set.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn,ort=info")),
        )
        .init();

    let raw_args: Vec<String> = std::env::args().collect();

    // Extract global flags before command dispatch.
    let use_npu = raw_args.iter().any(|a| a == "--npu");
    let use_rocm = raw_args.iter().any(|a| a == "--rocm");
    let use_cuda = raw_args.iter().any(|a| a == "--cuda");
    let use_openvino = raw_args.iter().any(|a| a == "--openvino");

    // Build a view of argv without backend flags for command matching.
    let backend_flags = ["--npu", "--rocm", "--cuda", "--openvino"];
    let args: Vec<&str> = raw_args
        .iter()
        .skip(1)
        .filter(|a| !backend_flags.contains(&a.as_str()))
        .map(|s| s.as_str())
        .collect();

    match args.first().copied() {
        Some("--version") | Some("-V") => {
            println!("breadmill {}", env!("CARGO_PKG_VERSION"));
        }
        Some("--fetch-model") | Some("fetch-model") => {
            if let Err(e) = fetch_model() {
                eprintln!("breadmill: {}", e);
                std::process::exit(1);
            }
        }
        Some("--reindex") | Some("reindex") => {
            if let Err(e) = run_daemon(true, use_npu, use_rocm, use_cuda, use_openvino) {
                eprintln!("breadmill: {}", e);
                std::process::exit(1);
            }
        }
        Some("query") => {
            let q = args.get(1).copied().unwrap_or("");
            if q.is_empty() {
                eprintln!("usage: breadmill query <text>");
                std::process::exit(1);
            }
            cli_query(q);
        }
        Some("status") => {
            cli_status();
        }
        None | Some("serve") | Some("--serve") => {
            if let Err(e) = run_daemon(false, use_npu, use_rocm, use_cuda, use_openvino) {
                eprintln!("breadmill: {}", e);
                std::process::exit(1);
            }
        }
        Some(cmd) => {
            eprintln!("breadmill: unknown command: {}", cmd);
            eprintln!(
                "usage: breadmill [serve|reindex|fetch-model|query <text>|status] [--npu|--rocm|--cuda|--openvino] [--version]"
            );
            std::process::exit(1);
        }
    }
}

// ---- Daemon -----------------------------------------------------------------

fn run_daemon(
    force_reindex: bool,
    use_npu: bool,
    use_rocm: bool,
    use_cuda: bool,
    use_openvino: bool,
) -> Result<(), String> {
    let config = breadsearch_shared::Config::load();
    let state_dir = breadsearch_shared::state_dir();
    let cache_dir = breadsearch_shared::cache_dir();
    let socket_path = breadsearch_shared::socket_path();
    let dim = config.model.dim;
    let snippet_len = config.search.snippet_len;
    let search_limit = config.search.limit;

    std::fs::create_dir_all(&state_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;

    // CLI flags always override config — otherwise an explicit --cuda/--npu on
    // the command line would silently lose to an unrelated `backend = "..."`
    // already sitting in config.toml, since that's whatever earlier branch a
    // fixed if/else-if priority order happened to check first.
    let backend_name = if use_npu {
        "npu"
    } else if use_rocm {
        "rocm"
    } else if use_cuda {
        "cuda"
    } else if use_openvino {
        "openvino"
    } else {
        config.model.backend.as_str()
    };

    let backend = match backend_name {
        "npu" => {
            eprintln!("breadmill: NPU backend selected");
            Backend::Npu { cache_dir: cache_dir.clone() }
        }
        "rocm" => {
            eprintln!("breadmill: ROCm backend selected");
            Backend::Rocm
        }
        "cuda" => {
            eprintln!("breadmill: CUDA backend selected");
            Backend::Cuda
        }
        "openvino" => {
            eprintln!("breadmill: OpenVINO backend selected");
            Backend::OpenVino { cache_dir: cache_dir.clone() }
        }
        _ => Backend::Cpu,
    };

    let store = Store::open(&state_dir, dim)?;
    let state = Arc::new(SharedState::new(store));

    // Load embedder if model files present
    let model_dir = model_dir(&cache_dir);
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if model_path.exists() && tokenizer_path.exists() {
        eprintln!("breadmill: loading model...");
        match OrtEmbedder::load(&model_path, &tokenizer_path, dim, backend) {
            Ok(embedder) => {
                *state.embedder.lock().unwrap() = Some(embedder);
                state.model_ready.store(true, Ordering::Relaxed);
                eprintln!("breadmill: model loaded");
            }
            Err(e) => eprintln!("breadmill: model load failed: {} — run --fetch-model", e),
        }
    } else {
        eprintln!(
            "breadmill: model files not found in {} — run: breadmill --fetch-model",
            model_dir.display()
        );
    }

    // Indexer runs in a background thread
    {
        let state_clone = Arc::clone(&state);
        let config_clone = config.clone();
        let state_dir_clone = state_dir.clone();

        std::thread::spawn(move || {
            let indexer = Indexer::new(state_clone, config_clone, state_dir_clone);
            if force_reindex {
                indexer.full_reindex();
            }
            indexer.run();
        });
    }

    // Server runs on the main thread (blocking)
    serve::run(&socket_path, Arc::clone(&state), snippet_len, search_limit);

    Ok(())
}

// ---- Model fetch ------------------------------------------------------------

fn fetch_model() -> Result<(), String> {
    let cache_dir = breadsearch_shared::cache_dir();
    let model_dir = model_dir(&cache_dir);
    std::fs::create_dir_all(&model_dir).map_err(|e| e.to_string())?;

    download_if_missing(MODEL_URL, &model_dir.join("model.onnx"))?;
    download_if_missing(TOKENIZER_URL, &model_dir.join("tokenizer.json"))?;

    eprintln!("breadmill: model files ready in {}", model_dir.display());
    Ok(())
}

fn download_if_missing(url: &str, dest: &Path) -> Result<(), String> {
    if dest.exists() {
        eprintln!("  already present: {}", dest.display());
        return Ok(());
    }

    eprintln!("  downloading {} ...", url);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(300))
        .build();

    let response = agent.get(url).call().map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;

    if bytes.is_empty() {
        return Err(format!("empty download from {}", url));
    }

    // Write atomically via temp file
    let tmp = dest.with_extension("tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dest).map_err(|e| e.to_string())?;

    eprintln!("  saved {} ({:.1} MB)", dest.display(), bytes.len() as f64 / 1_048_576.0);
    Ok(())
}

fn model_dir(cache_dir: &Path) -> PathBuf {
    cache_dir.join("models")
}

// ---- CLI helpers ------------------------------------------------------------

fn cli_query(query: &str) {
    let req = Request::Query {
        query: query.to_string(),
        limit: 10,
    };
    match breadsearch_shared::send_request(&req) {
        Ok(Response::Hits { hits }) => {
            if hits.is_empty() {
                println!("no results");
            }
            for (i, h) in hits.iter().enumerate() {
                println!(
                    "{:2}. {} ({:.3})\n    {}\n    {}\n",
                    i + 1,
                    h.title,
                    h.score,
                    h.path,
                    h.snippet.lines().next().unwrap_or(""),
                );
            }
        }
        Ok(Response::Error { message }) => eprintln!("error: {}", message),
        Ok(_) => eprintln!("unexpected response"),
        Err(e) => eprintln!("could not reach breadmill: {}", e),
    }
}

fn cli_status() {
    match breadsearch_shared::send_request(&Request::Status) {
        Ok(Response::StatusInfo(s)) => {
            println!("indexed:     {}", s.indexed);
            println!("pending:     {}", s.pending);
            println!("model ready: {}", s.model_ready);
        }
        Ok(_) => eprintln!("unexpected response"),
        Err(e) => eprintln!("could not reach breadmill: {}", e),
    }
}
