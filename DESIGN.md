# breadsearch + breadmill — semantic system-wide search for BOS

## Context
BOS/bread has no content search — only breadbox's app launcher (exact/fuzzy over `.desktop` files). The goal is a **semantic** "find anything by meaning" engine: a flagship, differentiating BOS feature that's also the *right* workload for the AMD XDNA NPU (small encoder model, compute-bound single forward pass, always-on background embedding — none of the bandwidth-bound problems that make LLMs a bad NPU fit).

Two components, mirroring the breadpad/breadman split and the breadbox/breadbox-sync precedent (GUI + background helper + shared lib in one repo):
- **breadmill** — always-on daemon: walks files → extracts text → chunks → embeds → vector index; answers queries over a Unix socket. ("mill grain into flour.")
- **breadsearch** — standalone GTK4 GUI, *forked from breadbox's UI*, that queries breadmill and shows ranked hits. ("sift the flour.")

breadbox stays a pure app launcher, unchanged.

## Decisions (confirmed with user)
- **Index scope (v1):** curated roots — `~/Documents`, `~/Projects` (notes/docs, not code yet), `~/.config/breadpad`. Extract `md, txt, org, pdf, odt, docx`. Skip binaries/images/build dirs/`.git`.
- **Repo layout:** one cargo workspace at `~/Projects/breadsearch/`.
- **Embedding model:** `nomic-embed-text-v1.5` (768-dim ONNX, ~550MB). Requires task prefixes: `search_document: ` for indexed chunks, `search_query: ` for queries; mean-pool + L2-normalize.
- **Compute:** CPU-first; NPU (XDNA via ONNX Runtime VitisAI EP) is a later backend swap, not a v1 dependency.

## Workspace layout
```
~/Projects/breadsearch/
  Cargo.toml              # [workspace] members = breadsearch-shared, breadmill, breadsearch
  bakery.toml             # bread package manifest (binaries: breadsearch, breadmill)
  config.example.toml     # ~/.config/breadsearch/config.toml template
  README.md
  breadsearch-shared/       # lib: XDG paths, config, IPC types + socket client
  breadmill/              # daemon bin
  breadsearch/              # GUI bin (forked breadbox UI)
```

## Component: breadsearch-shared (lib)
Model on `breadbox/breadbox-shared/src/lib.rs` (XDG helpers + serde/toml config).
- **Paths:** `config_dir()` → `~/.config/breadsearch`; `state_dir()` → `~/.local/state/breadsearch` (index); `cache_dir()` → `~/.cache/breadsearch` (models); `socket_path()` → `$XDG_RUNTIME_DIR/breadmill.sock`.
- **Config** (serde + `toml`): `[index] roots, extensions, max_file_mb`; `[search] limit, snippet_len`; `[model] name, dim`.
- **IPC types** (serde_json, newline-delimited JSON over the Unix socket):
  - Request: `Query { query: String, limit: usize }`, `Status`, `Reindex`.
  - Response: `Hits(Vec<Hit>)` where `Hit { title, path, snippet, score }`; `StatusInfo { indexed, pending, model_ready }`.
- **Socket client** helper used by the GUI (connect, send, read one response).

## Component: breadmill (daemon)
Pipeline, isolated behind small traits so each stage is swappable:
1. **Walk** — `ignore` crate (parallel, respects `.gitignore`) over configured roots; filter by extension + size.
2. **Extract** — `md/txt/org`: read directly; `pdf`: `pdf-extract`; `docx/odt`: unzip + strip XML (`zip` + `quick-xml`), best-effort.
3. **Chunk** — ~512-token windows with overlap; keep byte offsets for snippets.
4. **Embed** — `Embedder` trait. v1 impl: `ort` (ONNX Runtime 2.x, CPU EP) + `tokenizers` (HF) running nomic-embed-text-v1.5. Apply `search_document:`/`search_query:` prefixes, mean-pool, normalize.
5. **Store** — `rusqlite` for metadata (path, mtime, content-hash, chunk text/offsets) keyed by rowid + `usearch` (HNSW, 768-dim, cosine) for vectors keyed by the same id. Both persisted under `state_dir()`.
6. **Incremental** — on start, diff roots against sqlite (mtime+hash): embed new/changed, drop deleted. Then live-watch with `notify` (debounced) to re-embed on change.
7. **Serve** — `tokio` (or std threads) Unix-socket listener: `Query` → embed query → usearch top-k → join sqlite metadata → `Hits`. Also `Status`/`Reindex`.
- **Model fetch:** first run downloads `model.onnx` + `tokenizer.json` from HF into `cache_dir()/models/` (needs network once); `breadmill --fetch-model` to pre-fetch. Log clearly if absent.
- **Lifecycle:** systemd **user** service `breadmill.service` (pattern from `breadbox-sync.service` / breadd), `WantedBy=default.target`.

## Component: breadsearch (GUI) — fork of breadbox
Start from `breadbox/breadbox/src/main.rs`. **Reuse verbatim:** the gtk4-layer-shell overlay window (rename namespace/app-id to `breadsearch` / `com.breadway.breadsearch`), `SearchEntry` + `ScrolledWindow` + `ListBox`, ↑/↓/Enter/Esc handling, click-outside-to-close, PID-toggle (`breadsearch.pid`), and the theming path: `bread_theme::gtk::apply_shared()` + `apply_app_css(|| build_css(&load_palette()))` + user `style.css`. Pin `bread-theme` git tag `v0.2.8`, feature `gtk` (same as breadbox).
**Swap:**
- **Result source:** delete `load_sorted_entries`/`fuzzy_*`/`DesktopEntry`. On `search.connect_changed`, **debounce ~150ms** (`glib::timeout_add_local`) then query breadmill **off the UI thread** (`std::thread` + `glib::MainContext::channel`), clear the `ListBox`, append a row per `Hit`.
- **Row content:** title (filename/heading) + muted path + snippet line; filetype icon via `gio::content_type_guess` → `Image::from_gicon`. Extend `build_css` with a `.hit-snippet` class.
- **Action:** replace `do_launch` with open-file — `Enter`/row-activated → `xdg-open <path>`; `Ctrl+Enter` → open containing folder. Then close.

## Key crates
`ort` (ONNX Runtime), `tokenizers`, `usearch`, `rusqlite`, `ignore`, `notify`, `pdf-extract`, `zip`+`quick-xml`, `serde`/`serde_json`/`toml`, `gtk4` 0.11 + `gtk4-layer-shell` 0.8, `bread-theme` (git tag v0.2.8).

## Packaging & BOS integration (last phase — post-1.0, per earlier decision)
- `bakery.toml` (model on `breadbox/bakery.toml`): `binaries = ["breadsearch","breadmill"]`, system_deps for onnxruntime/gtk; `[[service]] unit="breadmill.service" enable=true`; `[config] dir="~/.config/breadsearch"`.
- BOS: add `breadsearch`+`breadmill` to `build-local.sh` `BREAD_BINS`; autostart `breadmill.service`; Hyprland keybind (e.g. `SUPER+slash`) → `breadsearch` in the skel `hyprland.lua`.
- Release: dual remotes (origin GitHub + forgejo), bakery index regen — per the bread release train.

## Phasing (de-risked: ship CPU, NPU later)
1. Scaffold workspace + `breadsearch-shared` (paths, config, IPC types, socket client).
2. `breadmill` CPU pipeline end-to-end (walk→extract→chunk→embed→store→serve) + `--reindex`/`--fetch-model` + systemd unit.
3. `breadsearch` GUI fork (socket query + xdg-open + theme).
4. Packaging (bakery, config.example, README) + BOS wiring.
5. **Later:** NPU `Embedder` impl (ort VitisAI/XDNA EP) — the go/no-go POC; pure backend swap.

## Verification
- `breadmill --fetch-model` then `--reindex` over a small test corpus; log embedded-chunk count; confirm `state_dir` index persists across restart.
- Query the socket directly (a `breadmill query "..."` subcommand or `socat`) and confirm semantically-relevant hits with sane scores for a concept query (not keyword).
- Launch `breadsearch`, type a *concept* (e.g. "tax stuff", "that suspend bug fix"), see relevant files ranked, `Enter` opens via xdg-open, `Ctrl+Enter` reveals folder, `Esc` closes; theme matches breadbox; hot-reloads on `bread-theme reload`.
- Edit/add/delete a file in a root → `notify` re-index → new content findable within seconds.

## Notes / risks
- nomic prefixes + mean-pool + normalize must match between index and query or recall collapses.
- `ort` linking: prefer the crate's downloaded/bundled ONNX Runtime to avoid version skew with Arch's `onnxruntime`.
- Office formats (docx/odt) are best-effort in v1; md/txt/org/pdf are the reliable path.
- GPU EPs (ROCm/CUDA) fail to register silently at the ONNX Runtime level and fall back to CPU — always check
  startup logs for `Successfully registered` before trusting a GPU build is actually accelerating. See
  [README: GPU backend notes](README.md#gpu-backend-notes) for the MIGraphX-vs-ROCMExecutionProvider distinction
  and the per-shape JIT-compile-and-cache behavior that matters for interactive query latency.
