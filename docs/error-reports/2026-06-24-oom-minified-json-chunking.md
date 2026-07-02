# Error Report: breadmill OOM loop on minified JSON `.txt` file

**Date:** 2026-06-24  
**Reporter:** breadway  
**Component:** `breadmill` (semantic search indexer)  
**Version:** 0.1.0 (`/home/breadway/Projects/breadsearch/breadmill`)  
**Severity:** Critical — repeated kernel OOM kills, ~28 GB RAM consumption, system instability

---

## Summary

`breadmill` enters a crash loop when indexing a 705 KB minified JSON game save file exposed as `.txt`. Word-based chunking produces ~486–616 KB chunks that are passed unbounded to the ONNX embedding model. ONNX Runtime then attempts multi-terabyte allocations in attention `MatMul`, the process balloons to ~28 GB RSS, and the kernel OOM killer terminates it. `Restart=on-failure` causes an immediate restart and the cycle repeats.

On 2026-06-24 this produced **40 embed errors** and **76 OOM kills** in journal logs within ~45 minutes.

---

## Environment

| Item | Value |
|------|-------|
| Host | Lenovo Yoga Slim 7 14AKP10 (`83JY`) |
| OS | BOS (Arch-based), kernel `7.0.12-arch1-1` |
| RAM | 32 GB |
| Swap | 4 GB zram |
| Model | `nomic-embed-text-v1.5` (768-dim, ONNX via `ort` 2.0.0-rc.12) |
| Service | `breadmill.service` (user systemd unit) |
| Config | `~/.config/breadsearch/config.toml` |

### Service unit

```ini
[Service]
Type=simple
ExecStart=%h/.cargo/bin/breadmill
Restart=on-failure
RestartSec=5
# No MemoryMax set (unlimited)
# OOMScoreAdjust=200 (default for user services — preferential OOM victim)
```

### Relevant config

```toml
[index]
roots = ["~/Documents", "~/.config/breadpad"]
extensions = ["md", "txt", "org", "pdf", "odt", "docx"]
max_file_mb = 5.0
```

---

## Triggering file

**Path:**  
`~/Documents/Gaming/Games/Spaceflight Simulator Game/Saving/Worlds/Bread/Persistent/Rockets.txt`

| Property | Value |
|----------|-------|
| Size | 721,601 bytes (705 KB) |
| Type | JSON array (11 rocket objects), stored as `.txt` |
| Lines | **1** (single-line minified JSON) |
| Whitespace-delimited "words" | 601 |
| Longest "word" | 68,068 characters |

The file is under `max_file_mb = 5.0` and has extension `.txt`, so it is indexed.

---

## Root cause

### 1. Word-based chunking breaks on minified JSON

`chunk::chunk_text()` splits on whitespace only (`chunk.rs`). For minified JSON, most structure is packed into very long tokens with few spaces.

Chunking with `words_per_chunk=400`, `overlap_words=80` (`indexer.rs:268`) yields:

| Chunk | Word range | Character length |
|-------|------------|------------------|
| 0 | 0–399 | **486,347** |
| 1 | 320–600 | **616,582** |

A "400-word chunk" is not ~400 natural-language words; it is hundreds of kilobytes of dense JSON.

### 2. No token/character limit before embedding

`embed::OrtEmbedder::embed_with_prefix()` tokenizes the full chunk and runs ONNX inference with no `max_length` truncation (`embed.rs`). The nomic model supports ~8192 tokens; these chunks are orders of magnitude larger when tokenized.

### 3. ONNX attention allocation explodes

Embedding fails inside ONNX Runtime:

```
Non-zero status code returned while running FusedMatMul node.
Name: '/encoder/layers.0/attn/MatMul/MatMulScaleFusion/'
Status Message: ... BFCArena::AllocateRawInternal ...
Failed to allocate memory for requested buffer of size 5525780084992
```

Requested buffer: **~5.1 TiB** (5,525,780,084,992 bytes).

Kernel also logs repeated allocation attempts before OOM:

```
__vm_enough_memory: pid: NNNN, comm: breadmill,
  bytes: 8796093026304 not enough memory for the allocation   (~8.0 TiB)
  bytes: 7916483514368 not enough memory for the allocation   (~7.2 TiB)
  bytes: 7124834848768 not enough memory for the allocation   (~6.5 TiB)
```

### 4. Restart loop amplifies damage

`Restart=on-failure` + `RestartSec=5` restarts breadmill immediately after each OOM kill. Each restart reloads the model, rescans `~/Documents`, hits the same file, and OOMs again. Peak memory per attempt: **~27–28 GB RSS** (`anon-rss:27867432kB`).

---

## Log excerpts

### Embed error (repeats on every restart)

```
breadmill: embed error for .../Rockets.txt: Non-zero status code returned while running FusedMatMul node.
Name:'/encoder/layers.0/attn/MatMul/MatMulScaleFusion/' Status Message:
  ... Failed to allocate memory for requested buffer of size 5525780084992
```

### OOM kill

```
oom-kill: ... task_memcg=.../breadmill.service, task=breadmill, pid=4515, uid=1000
Out of memory: Killed process 4515 (breadmill)
  total-vm:64081848kB, anon-rss:27867432kB, ... oom_score_adj:200
systemd[1649]: breadmill.service: Failed with result 'oom-kill'.
systemd[1649]: breadmill.service: Consumed ... 27.9G memory peak, 2.1G memory swap peak.
systemd[1649]: breadmill.service: Scheduled restart job, restart counter is at 8.
```

### Secondary noise (non-fatal)

```
breadmill: extract .../Novacana Info.txt: stream did not contain valid UTF-8
```

---

## Reproduction

1. Place a single-line minified JSON file (>400 KB, `.txt` extension) under a configured index root.
2. Start `breadmill serve` (or `systemctl --user start breadmill.service`).
3. Wait for initial scan to reach the file.
4. Observe embed errors, climbing RSS, and OOM kills in `journalctl --user -u breadmill.service -f`.

**Minimal reproducer characteristics:**

- Extension in `extensions` list (e.g. `txt`)
- File size `< max_file_mb`
- Content is minified JSON or other whitespace-sparse text on one line
- Results in chunks >> model `max_seq_len` when tokenized

---

## Impact

- **breadmill** unusable while the file is present in index roots
- **System-wide** memory pressure: swap exhaustion, unrelated process kills, journal/cache pressure
- Can be mistaken for unrelated storage or suspend issues when swap write errors appear in kernel log

---

## Suggested fixes

### Required (correctness)

1. **Truncate before tokenization** — cap input to model `max_seq_len` (8192 tokens for nomic-embed-text-v1.5) in `embed.rs`, with explicit logging when truncation occurs.

2. **Character-based chunk limits** — add `max_chunk_chars` (e.g. 8_000–32_000) independent of word count; split oversized chunks before embedding.

3. **Sanity-check chunk size** — refuse to embed chunks above a byte/token threshold; log and skip rather than calling ONNX.

### Recommended (resilience)

4. **Per-chunk error isolation** — on embed failure for one chunk, skip that chunk but do not retry the entire file in a tight loop; mark file as `failed` in SQLite.

5. **Service memory limit** — set `MemoryMax=4G` (or similar) on `breadmill.service` so a runaway embed cannot take the whole machine.

6. **Lower `OOMScoreAdjust`** — use `0` or negative so breadmill is not preferentially killed while still allowing limits.

7. **Backoff on repeated OOM** — `RestartSec=exponential` or stop after N OOM kills per file.

### Optional (UX)

8. **Exclude patterns** — config option for glob excludes (e.g. `**/Saving/**`, `**/*.json` even if renamed `.txt`).

9. **Detect minified JSON** — if `serde_json::from_str` succeeds on `.txt`, skip or pretty-print/chunk differently.

10. **`--version` flag** — aids bug reports (currently `unknown command: --version`).

---

## Workaround (immediate)

Exclude the game save directory from index roots in `~/.config/breadsearch/config.toml`:

```toml
[index]
roots = [
    "~/Documents",
    "~/.config/breadpad",
]
# Then add an exclude mechanism when available, OR temporarily narrow roots:
# roots = ["~/Documents/Creative", "~/.config/breadpad"]
```

Or stop the service until a fix is deployed:

```bash
systemctl --user stop breadmill.service
```

---

## Files involved

| File | Role |
|------|------|
| `breadmill/src/chunk.rs` | Whitespace word chunking — no char/token cap |
| `breadmill/src/indexer.rs:268` | `chunk_text(&text, 400, 80)` — hardcoded params |
| `breadmill/src/embed.rs` | No truncation before `tokenizer.encode()` / ONNX run |
| `breadmill/src/extract.rs` | Treats `.txt` as raw UTF-8 (JSON passes through) |
| `~/.config/breadsearch/config.toml` | `max_file_mb` only; no chunk/token limits |

---

## Related

- ONNX node: `/encoder/layers.0/attn/MatMul/MatMulScaleFusion/`
- Model: [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (max sequence length 8192)
