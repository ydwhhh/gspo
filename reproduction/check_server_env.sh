#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
CHECK_STATUS=0

echo "== Operating system =="
uname -a || CHECK_STATUS=1

echo
echo "== Python executable =="
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
"${PYTHON_BIN}" --version || CHECK_STATUS=1
which "${PYTHON_BIN}" || CHECK_STATUS=1

echo
echo "== NVIDIA GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || CHECK_STATUS=1
else
    echo "ERROR: nvidia-smi was not found." >&2
    CHECK_STATUS=1
fi

echo
echo "== Python packages and CUDA =="
"${PYTHON_BIN}" - "${REPO_DIR}" <<'PY'
import sys
from pathlib import Path

repo_dir = Path(sys.argv[1]).resolve()
failed = False

try:
    import torch

    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("ERROR: PyTorch cannot access an NVIDIA GPU.", file=sys.stderr)
        failed = True
except Exception as exc:
    print(f"ERROR: cannot import torch: {exc}", file=sys.stderr)
    failed = True

try:
    import ray

    print(f"ray.__version__: {ray.__version__}")
except Exception as exc:
    print(f"ERROR: cannot import ray: {exc}", file=sys.stderr)
    failed = True

try:
    import vllm

    print(f"vllm.__version__: {vllm.__version__}")
except Exception as exc:
    print(f"ERROR: cannot import vllm: {exc}", file=sys.stderr)
    failed = True

try:
    import verl

    verl_file = Path(verl.__file__).resolve()
    expected_dir = (repo_dir / "verl").resolve()
    print(f"verl.__file__: {verl_file}")
    try:
        verl_file.relative_to(expected_dir)
    except ValueError:
        print(
            f"ERROR: imported veRL is not from this repository.\n"
            f"  imported: {verl_file}\n"
            f"  expected under: {expected_dir}",
            file=sys.stderr,
        )
        failed = True
except Exception as exc:
    print(f"ERROR: cannot import verl: {exc}", file=sys.stderr)
    failed = True

raise SystemExit(1 if failed else 0)
PY
PYTHON_STATUS=$?
if (( PYTHON_STATUS != 0 )); then
    CHECK_STATUS=1
fi

echo
if (( CHECK_STATUS != 0 )); then
    echo "Server environment check FAILED. No environment changes were made." >&2
    exit 1
fi

echo "Server environment check PASS."
