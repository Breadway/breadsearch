#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${CARGO_TARGET_DIR:-${ROOT}/target}/debug/breadmill"
if [[ "${1:-}" == "--bin" ]]; then BIN="$2"; shift 2; fi
CPU_ORT="${BREADMILL_TEST_CPU_ORT:-}"
[[ -x "$BIN" ]] || { echo "missing $BIN; build breadmill first" >&2; exit 1; }

tmp="$(mktemp -d /tmp/breadmill-runtime-XXXXXX)"
pid=""
cleanup() {
  if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  rm -rf "$tmp"
}
trap cleanup EXIT

MODEL_DIR="${BREADMILL_TEST_MODEL_DIR:-$tmp/model}"
mkdir -p "$MODEL_DIR"
python3 - "$MODEL_DIR" <<'PY'
import hashlib
import os
import pathlib
import sys
import time
import urllib.request

root = pathlib.Path(sys.argv[1])
files = {
    "model.onnx": (
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/e9b6763023c676ca8431644204f50c2b100d9aab/onnx/model.onnx",
        "147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965",
    ),
    "tokenizer.json": (
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/e9b6763023c676ca8431644204f50c2b100d9aab/tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
}
for name, (url, expected) in files.items():
    dest = root / name
    if not dest.is_file():
        temp = dest.with_suffix(dest.suffix + ".tmp")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, temp.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                os.replace(temp, dest)
                break
            except Exception:
                temp.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"{name} SHA-256 mismatch: {digest}")
    print(f"{name}: SHA-256 verified")
PY
echo '147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965  '"$MODEL_DIR/model.onnx" | sha256sum -c -
echo 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66  '"$MODEL_DIR/tokenizer.json" | sha256sum -c -

run_case() {
  local name="$1" dylib="$2" expected="$3" backend_flag="${4:-}"
  local home="$tmp/$name/home" cfg="$tmp/$name/config" cache="$tmp/$name/cache" state="$tmp/$name/state" runtime="$tmp/$name/runtime"
  mkdir -p "$home" "$cfg" "$cache/breadsearch/models" "$state" "$runtime"
  cp "$MODEL_DIR/model.onnx" "$MODEL_DIR/tokenizer.json" "$cache/breadsearch/models/"
  mkdir -p "$cfg/breadsearch"
  cat > "$cfg/breadsearch/config.toml" <<'EOF'
[index]
roots = []
[model]
backend = "cpu"
EOF
  if [[ "$dylib" == "<unset>" ]]; then unset ORT_DYLIB_PATH || true; else export ORT_DYLIB_PATH="$dylib"; fi
  local -a command=("$BIN" serve)
  [[ -z "$backend_flag" ]] || command+=("$backend_flag")
  HOME="$home" XDG_CONFIG_HOME="$cfg" XDG_CACHE_HOME="$cache" XDG_STATE_HOME="$state" XDG_RUNTIME_DIR="$runtime" "${command[@]}" >"$tmp/$name.log" 2>&1 &
  pid=$!
  local socket="$runtime/breadmill.sock"
  for _ in {1..300}; do [[ -S "$socket" ]] && break; sleep 0.1; done
  [[ -S "$socket" ]] || { cat "$tmp/$name.log" >&2; echo "socket did not start: $name" >&2; return 1; }
  local response=""
  local matched=0
  for _ in {1..300}; do
    response="$(python3 - "$socket" <<'PY'
import json, socket, sys
s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.sendall(b'{"type":"status"}\n')
print(s.makefile().readline().strip())
PY
)"
    if grep -q 'thread .* panicked' "$tmp/$name.log"; then
      cat "$tmp/$name.log" >&2
      echo "loader panicked: $name" >&2
      return 1
    fi
    if python3 - "$response" "$expected" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); expected=sys.argv[2]
ready=d.get('model_ready',False); error=d.get('model_error') or ''
if expected == 'ready': raise SystemExit(0 if ready else 1)
if expected == 'unavailable': raise SystemExit(0 if (not ready and 'ONNX Runtime' in error) else 1)
PY
    then matched=1; break; fi
    sleep 0.1
  done
  [[ "$matched" == 1 ]] || { echo "expected model state not reached: $name" >&2; cat "$tmp/$name.log" >&2; return 1; }
  echo "$response" | python3 -m json.tool
  if [[ "$expected" == "ready" ]]; then
    python3 - "$response" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); assert d['model_ready'] is True, d
PY
  else
    python3 - "$response" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); assert d['model_ready'] is False, d; assert 'ONNX Runtime' in (d.get('model_error') or ''), d
PY
  fi
  kill "$pid"; wait "$pid" 2>/dev/null || true; pid=""
}

run_case bundled-unset '<unset>' ready
run_case missing-override "$tmp/does-not-exist.so" unavailable
printf 'not an ORT shared library\n' > "$tmp/corrupt.so"
run_case corrupt-override "$tmp/corrupt.so" unavailable
if [[ -n "$CPU_ORT" ]]; then run_case supplied-cpu "$CPU_ORT" ready --cuda; else echo "SKIP supplied CPU ORT: set BREADMILL_TEST_CPU_ORT"; fi
