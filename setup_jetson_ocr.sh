#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python3"
  fi
fi
PADDLE_WHEEL="${PADDLE_WHEEL:-}"

log() {
  printf '[jetson-ocr] %s\n' "$1"
}

log "Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Failed to create the virtual environment. Install python3-venv and try again." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

log "Checking for system dependencies"
sudo apt-get update && sudo apt-get install -y patchelf

log "Upgrading packaging tools with compatibility versions"
python -m pip install --upgrade pip
python -m pip install "setuptools==58.2.0" wheel

log "Installing NumPy 1.x (to prevent binary incompatibility)"
python -m pip install "numpy<2.0.0"

log "Installing Python requirements"
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ -n "${PADDLE_WHEEL}" ]]; then
  if [[ ! -f "${PADDLE_WHEEL}" ]]; then
    echo "PADDLE_WHEEL was set but the file does not exist: ${PADDLE_WHEEL}" >&2
    exit 1
  fi

  log "Installing PaddlePaddle GPU wheel from ${PADDLE_WHEEL}"
  python -m pip install "${PADDLE_WHEEL}"
else
  cat <<'EOF'

PaddlePaddle GPU for Jetson is not installed yet.

Install the ARM64/L4T wheel that matches your JetPack version, then re-run this script with:

  export PADDLE_WHEEL=/path/to/paddlepaddle_gpu-<version>-cp310-cp310-linux_aarch64.whl
  ./setup_jetson_ocr.sh

Official references:
  https://www.paddlepaddle.org.cn/install/quick
  https://www.paddlepaddle.org.cn/documentation/docs/en/install/quick_start.html
EOF
fi

python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("paddle")
if spec is None:
    print("\n[jetson-ocr] Paddle is still missing. Install the Jetson GPU wheel before running main.py.\n")
else:
    import paddle
    print(f"\n[jetson-ocr] Paddle detected: {paddle.__version__}\n")
PY

log "Setup complete"