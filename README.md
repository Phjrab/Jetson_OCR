# Jetson OCR for Jetson Orin Nano

High-accuracy real-time OCR for NVIDIA Jetson Orin Nano using PaddleOCR, tuned for Korean and English book pages.

## What This Project Uses

- JetPack 6.0+ / Ubuntu 22.04
- Python virtual environment created with `--system-site-packages`
- JetPack-provided OpenCV and CUDA
- PaddleOCR with GPU acceleration enabled
- UTF-8-safe OCR overlays rendered through PIL with a Korean-capable font

## Files

- [setup_jetson_ocr.sh](setup_jetson_ocr.sh)
- [main.py](main.py)
- [requirements.txt](requirements.txt)

## Setup

1. Make the setup script executable:

```bash
chmod +x setup_jetson_ocr.sh
```

2. Run the setup script from the project root:

```bash
./setup_jetson_ocr.sh
```

3. Install the Jetson-specific PaddlePaddle GPU wheel.

PaddlePaddle does not ship a single universal `paddlepaddle-gpu` wheel for Jetson on PyPI. You need the ARM64 / L4T build that matches your JetPack, CUDA, Python, and Ubuntu version.

**Option A: Pre-built Wheel from GitHub Releases (Recommended)**

Optimized wheel for Jetson Orin Nano (Compute Capability 8.7, CUDA 12.6, Python 3.10):

```bash
cd /home/jetson_orin_nano/project/Jetson_OCR
source .venv/bin/activate
wget https://github.com/Phjrab/Jetson_OCR/releases/download/v3.3.0-jetson-orin-nano-8.7/paddlepaddle_gpu-3.3.0.dev20251226-cp310-cp310-linux_aarch64.whl
pip install ./paddlepaddle_gpu-3.3.0.dev20251226-cp310-cp310-linux_aarch64.whl
```

Alternatively, if the wheel is already in the project directory:

```bash
source .venv/bin/activate
pip install ./paddlepaddle_gpu-3.3.0.dev20251226-cp310-cp310-linux_aarch64.whl
```

**Option B: Manual Download**

Official install references:

- PaddlePaddle install guide: https://www.paddlepaddle.org.cn/install/quick
- PaddlePaddle documentation: https://www.paddlepaddle.org.cn/documentation/docs/en/install/quick_start.html

Manual install flow:

```bash
# Example: install the wheel you downloaded for Jetson/L4T
pip install /path/to/paddlepaddle_gpu-<version>-cp310-cp310-linux_aarch64.whl
```

If you already have the wheel file, you can also use the environment variable before running the setup script:

```bash
export PADDLE_WHEEL=/path/to/paddlepaddle_gpu-3.3.0.dev20251226-cp310-cp310-linux_aarch64.whl
./setup_jetson_ocr.sh
```

## Performance Tuning

Enable maximum Jetson performance before launching OCR:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Run

```bash
source .venv/bin/activate
python main.py
```

## Runtime Notes

- Default camera source is `/dev/video0`.
- The script uses 1920x1080 capture by default for accuracy.
- Korean overlay rendering depends on a Korean-capable font such as Noto Sans CJK or Nanum Gothic.
- `enable_mkldnn` is disabled because this is an ARM Jetson target, not x86.
- TensorRT support is exposed as a placeholder flag and only works if your PaddlePaddle wheel and inference stack support it.

## Accuracy-Oriented Model Guidance

For best results on books and printed text, keep the default high-accuracy configuration and, if needed, point PaddleOCR to a PP-OCRv4 or server-grade model directory using the command-line options in `main.py`.

## Troubleshooting

- If Korean text appears as boxes or garbled output, install `fonts-noto-cjk` or `fonts-nanum`.
- If `paddle` cannot be imported, the Jetson ARM64/L4T GPU wheel has not been installed yet.
- If the camera does not open, confirm that `/dev/video0` exists and another application is not using it.