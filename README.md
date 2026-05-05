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

**Option C: Build from Source (Advanced)**

For custom optimizations or if pre-built wheels are unavailable, compile PaddlePaddle 3.3.0 directly on Jetson Orin Nano.

**Prerequisites:**
- 8+ GB free disk space
- Swap memory already configured
- CUDA 12.6 and cuDNN 9.3.0 installed

**Step 1: Prepare Build Environment**

```bash
# 1. Deactivate conda to avoid conflicts
conda deactivate

# 2. Install system dependencies for packaging
sudo apt-get update
sudo apt-get install -y patchelf libGL

# 3. Set up Python virtual environment with pinned versions
source ~/project/Jetson_OCR/.venv/bin/activate
pip install --upgrade pip
pip install "setuptools==58.2.0" "numpy<2.0.0" wheel
```

**Step 2: Configure CMake (Orin Nano Optimized)**

```bash
cd ~/Paddle
mkdir -p build && cd build

cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_GPU=ON \
    -DWITH_TESTING=OFF \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="8.7" \
    -DWITH_ARM=ON \
    -DWITH_AVX=OFF \
    -DWITH_MKL=OFF \
    -DWITH_MKLDNN=OFF \
    -DWITH_TENSORRT=OFF \
    -DWITH_NCCL=OFF \
    -DWITH_DISTRIBUTE=OFF \
    -DWITH_NVJPEG=OFF \
    -DCMAKE_CUDA_FLAGS="-U__ARM_NEON -DEIGEN_DONT_VECTORIZE=1" \
    -DPYTHON_EXECUTABLE=$(which python3) \
    2>&1 | tee cmake_output.log
```

**Step 3: Compile (Adaptive Core Strategy)**

To avoid "Killed" errors from memory exhaustion, use a two-phase compilation strategy:

```bash
# Phase 1: Start with 6 cores for speed
ninja -j6 2>&1 | tee build_output.log

# Phase 2: If "Killed" error occurs around build step ~700,
# restart with reduced cores to escape heavy Fused Kernel compilation
ninja -j2 2>&1 | tee -a build_output.log
```

**Step 4: Verify and Install**

```bash
# 1. Confirm wheel generation
ls -lh ~/Paddle/build/python/dist/

# 2. Install the generated wheel
pip install ~/Paddle/build/python/dist/paddlepaddle_gpu-*.whl --force-reinstall

# 3. Validate GPU recognition
python3 -c "import paddle; paddle.utils.run_check()"
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

## Dependency Requirements

| Package | Version | Notes |
|---------|---------|-------|
| **paddlepaddle-gpu** | 3.0.0b1 ~ 3.3.0.dev | **[CRITICAL]** Direct wheel build required. Never use `pip install paddlepaddle-gpu`. Install from [GitHub Releases](https://github.com/Phjrab/Jetson_OCR/releases/tag/v3.3.0-jetson-orin-nano-8.7) only. |
| paddleocr | >=2.8.1 | OCR core library |
| **numpy** | >=1.23.0, <2.0.0 | **[CRITICAL]** NumPy 2.x breaks binary compatibility. Pinned to 1.x range. |
| **setuptools** | ==58.2.0 | **[IMPORTANT]** Latest versions cause distutils build errors on ARM. Pinned in setup script. |
| opencv-python | >=4.6.0 | Use standard version, not headless variant for Jetson. |
| Pillow | >=10.0.0 | Image processing and text overlay rendering. |
| shapely | >=2.0.0 | OCR bounding box operations. |
| pyclipper | >=1.3.0 | Text box expansion/contraction operations. |

## Troubleshooting

- **"ModuleNotFoundError: No module named 'paddle'"** → Wheel not installed. Download from [Releases](https://github.com/Phjrab/Jetson_OCR/releases) and run: `pip install ./paddlepaddle_gpu-3.3.0.dev20251226-cp310-cp310-linux_aarch64.whl`
- **"ImportError: numpy.*.so"** → NumPy 2.x detected. Run: `pip install 'numpy<2.0.0'` and reinstall paddle wheel.
- **"distutils not found"** → setuptools version mismatch. The setup script pins it to 58.2.0; if not applied, run: `pip install setuptools==58.2.0`
- **Korean text appears as boxes or garbled** → Install a Korean-capable font: `sudo apt-get install fonts-noto-cjk fonts-nanum`
- **Camera does not open** → Confirm `/dev/video0` exists and no other app is using it: `ls -l /dev/video0`