#!/usr/bin/env python3
"""
Test VitisAI EP with the quantized nomic-embed-text-v1.5 model.
Checks whether the NPU VAIML pass achieves meaningful GOPs coverage.
"""

import os
import json
import time
import numpy as np
from pathlib import Path

MODEL_PATH = Path.home() / ".cache/breadsearch/models/model_quantized_static.onnx"
CACHE_DIR = Path.home() / ".cache/breadsearch/npu/nomic-quantized-static"
VAIP_CONFIG = Path.home() / ".config/breadsearch/vaip_config.json"
RYZEN_AI_LIB = Path.home() / ".local/share/ryzen-ai-1.7.1/lib"

# libvaiml.so must be discoverable
os.environ["RYZEN_AI_INSTALLATION_PATH"] = str(RYZEN_AI_LIB)
os.environ["LD_LIBRARY_PATH"] = str(RYZEN_AI_LIB) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not VAIP_CONFIG.exists():
    print(f"ERROR: vaip_config.json not found at {VAIP_CONFIG}")
    print("Check breadmill embed.rs for the find_vaip_config() paths")
    exit(1)

print(f"Testing quantized model: {MODEL_PATH}")
print(f"Cache dir: {CACHE_DIR}")
print(f"VAIP config: {VAIP_CONFIG}")
print(f"RYZEN_AI_INSTALLATION_PATH: {os.environ['RYZEN_AI_INSTALLATION_PATH']}")

import onnxruntime as ort

providers = [
    ("VitisAIExecutionProvider", {
        "config_file": str(VAIP_CONFIG),
        "cacheDir": str(CACHE_DIR),
        "cacheKey": "nomic-quantized-static",
    }),
    "CPUExecutionProvider",
]

print("\nCreating InferenceSession with VitisAI EP...")
t0 = time.time()
try:
    sess = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    t1 = time.time()
    print(f"Session created in {t1-t0:.1f}s")
    print(f"Active providers: {sess.get_providers()}")
except Exception as e:
    print(f"ERROR creating session: {e}")
    exit(1)

# Check for the VAIML pass summary
summary_path = CACHE_DIR / "nomic-quantized" / "preliminary-vaiml-pass-summary.txt"
if not summary_path.exists():
    # Try variations
    for p in CACHE_DIR.rglob("preliminary-vaiml-pass-summary.txt"):
        summary_path = p
        break

if summary_path.exists():
    print(f"\n--- VAIML Pass Summary ---")
    print(summary_path.read_text())
else:
    print(f"\nNo VAIML summary found at {summary_path}")
    print("Files in cache dir:")
    for f in CACHE_DIR.rglob("*"):
        if f.is_file():
            print(f"  {f}")

# Run a quick inference test
print("\nRunning inference test...")
seq_len = 128
dummy_ids = np.ones((1, seq_len), dtype=np.int64)
dummy_mask = np.ones((1, seq_len), dtype=np.int64)
dummy_types = np.zeros((1, seq_len), dtype=np.int64)

input_names = [inp.name for inp in sess.get_inputs()]
print(f"Input names: {input_names}")

feed = {}
for name in input_names:
    if "type" in name:
        feed[name] = dummy_types
    elif "mask" in name:
        feed[name] = dummy_mask
    else:
        feed[name] = dummy_ids

t0 = time.time()
outputs = sess.run(None, feed)
t1 = time.time()
print(f"Inference completed in {(t1-t0)*1000:.1f}ms")
print(f"Output shape: {outputs[0].shape}")
