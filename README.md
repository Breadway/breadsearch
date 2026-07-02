# breadsearch

Semantic document search for Bread OS. Type a concept — not a keyword — and get ranked hits from your documents.

Two binaries in one Cargo workspace:

- **breadmill** — background daemon. Walks configured directories, extracts text, chunks and embeds documents with [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768-dim ONNX), stores vectors in an HNSW index (usearch) backed by SQLite metadata, and serves queries over a Unix socket. Watches for filesystem changes and re-indexes incrementally.
- **breadsearch** — GTK4 overlay GUI. Queries breadmill via the Unix socket and shows ranked results. Press Enter to open a file, Ctrl+Enter to reveal its folder, Esc to close. Bind it to a hotkey (e.g. Super+/) and invoke it as a toggle.

## Build

System dependencies: `gtk4`, `gtk4-layer-shell`, `librsvg`.

```
cargo build --release
```

The `ort` crate downloads a bundled ONNX Runtime at build time (no system `onnxruntime` needed for the default CPU build).

Optional features:

| Feature | What it adds |
|---------|-------------|
| `npu`   | AMD XDNA NPU via VitisAI ONNX Runtime EP (requires Ryzen AI SDK) |
| `rocm`  | AMD iGPU via ROCm ONNX Runtime EP |

```
# NPU build
cargo build --release -p breadmill --features npu
```

## Setup

**1. Fetch the embedding model** (~550 MB, downloaded once from Hugging Face):

```
breadmill fetch-model
```

Model files are stored in `~/.cache/breadsearch/models/`.

**2. Enable the systemd user service:**

```
cp packaging/breadmill.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now breadmill
```

Or run it directly: `breadmill serve` (or just `breadmill`).

**3. Copy the example config** (optional — built-in defaults are used otherwise):

```
mkdir -p ~/.config/breadsearch
cp config.example.toml ~/.config/breadsearch/config.toml
```

## breadsearch (GUI)

```
breadsearch
```

Invoking again while running closes the window (PID-toggle). Bind to a hotkey in your compositor config.

Results show: filename/title, full path, and a snippet from the matching chunk. Score is cosine similarity as a percentage.

User CSS overrides go in `~/.config/breadsearch/style.css`.

## breadmill (daemon / CLI)

```
# Start the daemon (also the default when called with no arguments)
breadmill serve

# Force a full re-index from scratch
breadmill reindex

# Download model files
breadmill fetch-model

# Query from the terminal
breadmill query "tax stuff"

# Show daemon status (chunks indexed, pending, model ready)
breadmill status

# Backend flags (requires the matching Cargo feature)
breadmill --npu
breadmill --rocm
```

## Config

`~/.config/breadsearch/config.toml` — all keys are optional; built-in defaults are shown.

```toml
[index]
roots        = ["~/Documents", "~/Projects", "~/.config/breadpad"]
extensions   = ["md", "txt", "org", "pdf", "odt", "docx"]
excludes     = []          # paths to skip (prefix match)
max_file_mb  = 10.0

[search]
limit        = 10          # max results per query
snippet_len  = 200         # max characters in result snippet

[model]
name         = "nomic-embed-text-v1.5"
dim          = 768
backend      = "cpu"       # "cpu", "npu", or "rocm"
```

`roots` and `excludes` support `~/` expansion. The index respects `.gitignore` files found during the walk.

### NPU backend

Set `backend = "npu"` in config (or pass `--npu`) when running a build compiled with `--features npu`. breadmill looks for the VitisAI EP config file in this order:

1. `$VAIP_CONFIG`
2. `~/.config/breadsearch/vaip_config.json`
3. `~/.local/share/ryzen-ai-1.7.1/voe-4.0-linux_x86_64/vaip_config.json`
4. `/etc/vaip_config.json`
5. `/opt/xilinx/vaip_config.json`

## Runtime paths

| Purpose | Path |
|---------|------|
| Config  | `~/.config/breadsearch/` |
| Index (SQLite + HNSW) | `~/.local/state/breadsearch/` |
| Model cache | `~/.cache/breadsearch/models/` |
| Unix socket | `$XDG_RUNTIME_DIR/breadmill.sock` |
