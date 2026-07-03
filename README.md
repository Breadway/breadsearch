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
| `rocm`  | AMD iGPU via the MIGraphX ONNX Runtime EP (ROCm-backed) |
| `cuda`  | NVIDIA GPU via the CUDA ONNX Runtime EP |
| `full`  | All three of the above in one binary |

```
# NPU build
cargo build --release -p breadmill --features npu

# ROCm (AMD iGPU/dGPU) build
cargo build --release -p breadmill --features rocm

# CUDA (NVIDIA GPU) build
cargo build --release -p breadmill --features cuda

# All backends in one binary (what the release build ships)
cargo build --release -p breadmill --features full
```

`rocm`/`cuda`/`npu` all use `ort`'s `load-dynamic` mode: at runtime, breadmill
dlopens whatever `libonnxruntime.so` the dynamic linker resolves (or
`ORT_DYLIB_PATH` if set). GPU acceleration only works if that ONNX Runtime
build actually has the matching execution provider compiled in — breadmill
logs a clear `Successfully registered` / `not enabled in this build` line for
this at startup (see [GPU backend notes](#gpu-backend-notes) below).

Because all three are dlopen-based, `full` doesn't require the NPU/ROCm/CUDA
toolkits to be installed at build time — only at run time, and only for
whichever single backend you actually select via `--npu`/`--rocm`/`--cuda`
or `backend` in config.toml. The **released binaries are built with
`full`**: same binary works CPU-only out of the box, and picks up NPU/ROCm/CUDA
acceleration on a machine that has the matching ONNX Runtime available,
without needing a different download. An explicit `--npu`/`--rocm`/`--cuda`
flag always overrides `backend` in config.toml, not the other way around.

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
breadmill --cuda
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
backend      = "cpu"       # "cpu", "npu", "rocm", or "cuda"
```

`roots` and `excludes` support `~/` expansion. The index respects `.gitignore` files found during the walk.

### NPU backend

Set `backend = "npu"` in config (or pass `--npu`) when running a build compiled with `--features npu`. breadmill looks for the VitisAI EP config file in this order:

1. `$VAIP_CONFIG`
2. `~/.config/breadsearch/vaip_config.json`
3. `~/.local/share/ryzen-ai-1.7.1/voe-4.0-linux_x86_64/vaip_config.json`
4. `/etc/vaip_config.json`
5. `/opt/xilinx/vaip_config.json`

### GPU backend notes

Both `rocm` and `cuda` need a system ONNX Runtime that was actually built with
the matching execution provider — the crate's own downloaded binary is CPU-only.
Point `ORT_DYLIB_PATH` at one, or install a distro package that provides
`libonnxruntime.so` with the EP baked in and let the dynamic linker find it.

**ROCm (`--rocm` / `backend = "rocm"`)** targets ONNX Runtime's **MIGraphX**
execution provider, not the classic `ROCMExecutionProvider`. Distro
ROCm-enabled ONNX Runtime packages (e.g. Arch's `onnxruntime-rocm`) are
commonly built with `--use_migraphx` rather than `--use_rocm`, so this is the
EP that's actually available in practice; the classic ROCm EP needs a bespoke
`--use_rocm` build most distros don't package. Startup logs a
`Successfully registered `MIGraphXExecutionProvider`` line when it's really
active — check for it if in doubt, since a failed GPU EP registration falls
back to CPU silently at the ONNX Runtime level (breadmill's own log line is
only a statement of intent, not a confirmation).

MIGraphX JIT-compiles the model per distinct input sequence length and caches
the compiled kernel to disk (each compile takes ~60–120s and produces a
~500MB `.mxr` file). Set `ORT_MIGRAPHX_MODEL_CACHE_PATH=/path/to/cache` so
that cost is paid once per shape instead of on every daemon restart. Because
query text length varies, expect an occasional multi-second stall the first
time a new token length is seen — fine for background document indexing,
noticeable for interactive query embedding.

**CUDA (`--cuda` / `backend = "cuda"`)** targets the standard
`CUDAExecutionProvider` and needs a CUDA-enabled ONNX Runtime + a working
CUDA/cuDNN install. Unverified on real NVIDIA hardware in this repo — only
compile-checked, since development happened on an AMD-only machine.

## Runtime paths

| Purpose | Path |
|---------|------|
| Config  | `~/.config/breadsearch/` |
| Index (SQLite + HNSW) | `~/.local/state/breadsearch/` |
| Model cache | `~/.cache/breadsearch/models/` |
| Unix socket | `$XDG_RUNTIME_DIR/breadmill.sock` |
