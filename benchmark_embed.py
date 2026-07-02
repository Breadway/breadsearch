#!/usr/bin/env python3
"""
Benchmark CPU (float32) vs CPU (int8 quantized) vs VitisAI (int8 quantized static).
"""

import os
import time
import numpy as np
from pathlib import Path

FLOAT32_MODEL = Path.home() / ".cache/breadsearch/models/model.onnx"
QUANT_MODEL   = Path.home() / ".cache/breadsearch/models/model_quantized.onnx"
STATIC_MODEL  = Path.home() / ".cache/breadsearch/models/model_quantized_static.onnx"
CACHE_DIR     = Path.home() / ".cache/breadsearch/npu/nomic-quantized-static"
VAIP_CONFIG   = Path.home() / ".config/breadsearch/vaip_config.json"
RYZEN_AI_LIB  = Path.home() / ".local/share/ryzen-ai-1.7.1/lib"

os.environ["RYZEN_AI_INSTALLATION_PATH"] = str(RYZEN_AI_LIB)
os.environ["LD_LIBRARY_PATH"] = str(RYZEN_AI_LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import onnxruntime as ort

WARMUP = 2
RUNS   = 10
SEQ_FLOAT = 512    # dynamic model accepts any seq len
SEQ_STATIC = 512   # static model locked to this

def make_input(seq_len: int):
    ids   = np.ones((1, seq_len), dtype=np.int64)
    mask  = np.ones((1, seq_len), dtype=np.int64)
    types = np.zeros((1, seq_len), dtype=np.int64)
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": mask}

def time_session(sess: ort.InferenceSession, feed: dict, n: int) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append(time.perf_counter() - t0)
    return times

def stats(times):
    arr = np.array(times)
    return arr.mean(), arr.min(), arr.max()

print("=" * 60)
print("BENCHMARK: nomic-embed-text-v1.5 embedding speed")
print("=" * 60)

# ── 1. float32 CPU ────────────────────────────────────────────
print("\n[1] float32 model — CPU EP")
sess = ort.InferenceSession(str(FLOAT32_MODEL), providers=["CPUExecutionProvider"])
feed = make_input(SEQ_FLOAT)
for _ in range(WARMUP): sess.run(None, feed)
times = time_session(sess, feed, RUNS)
mean, lo, hi = stats(times)
print(f"    seq={SEQ_FLOAT}  mean={mean*1000:.0f}ms  min={lo*1000:.0f}ms  max={hi*1000:.0f}ms  ({RUNS} runs)")

# ── 2. int8 quantized CPU (dynamic) ───────────────────────────
print("\n[2] int8 quantized model (dynamic shapes) — CPU EP")
sess = ort.InferenceSession(str(QUANT_MODEL), providers=["CPUExecutionProvider"])
feed = make_input(SEQ_FLOAT)
for _ in range(WARMUP): sess.run(None, feed)
times = time_session(sess, feed, RUNS)
mean2, lo2, hi2 = stats(times)
print(f"    seq={SEQ_FLOAT}  mean={mean2*1000:.0f}ms  min={lo2*1000:.0f}ms  max={hi2*1000:.0f}ms  ({RUNS} runs)")
print(f"    Speedup vs float32: {mean/mean2:.2f}x")

# ── 3. int8 quantized static — VitisAI EP ─────────────────────
print("\n[3] int8 quantized model (static shapes) — VitisAI EP (NPU+CPU)")
providers = [
    ("VitisAIExecutionProvider", {
        "config_file": str(VAIP_CONFIG),
        "cacheDir": str(CACHE_DIR),
        "cacheKey": "nomic-quantized-static",
    }),
    "CPUExecutionProvider",
]
print("    Loading session (should be fast — already compiled)...")
t_load = time.perf_counter()
sess_npu = ort.InferenceSession(str(STATIC_MODEL), providers=providers)
print(f"    Load time: {time.perf_counter()-t_load:.1f}s")
feed_static = make_input(SEQ_STATIC)
for _ in range(WARMUP): sess_npu.run(None, feed_static)
times_npu = time_session(sess_npu, feed_static, RUNS)
mean3, lo3, hi3 = stats(times_npu)
print(f"    seq={SEQ_STATIC}  mean={mean3*1000:.0f}ms  min={lo3*1000:.0f}ms  max={hi3*1000:.0f}ms  ({RUNS} runs)")
print(f"    Speedup vs float32: {mean/mean3:.2f}x")
print(f"    Speedup vs int8 CPU: {mean2/mean3:.2f}x")

print("\n" + "=" * 60)
print("SUMMARY")
print(f"  float32 CPU     : {mean*1000:.0f}ms/inference")
print(f"  int8 CPU        : {mean2*1000:.0f}ms/inference  ({mean/mean2:.2f}x)")
print(f"  int8 VitisAI    : {mean3*1000:.0f}ms/inference  ({mean/mean3:.2f}x)")
print("=" * 60)
