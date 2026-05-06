from __future__ import annotations

import argparse
import logging  # 추가됨: 로깅 제어용
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import paddle  # 추가됨: GPU 수동 할당용
from PIL import Image, ImageDraw, ImageFont

KOREAN_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)

@dataclass(frozen=True)
class OCRRuntimeConfig:
    camera_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    lang: str = "korean"
    use_gpu: bool = True
    use_textline_orientation: bool = True
    enable_mkldnn: bool = False
    ir_optim: bool = True
    use_tensorrt: bool = False
    det_model_dir: str | None = None
    rec_model_dir: str | None = None
    cls_model_dir: str | None = None
    font_path: str | None = None
    font_size: int = 24

def parse_args() -> OCRRuntimeConfig:
    parser = argparse.ArgumentParser(description="Real-time PaddleOCR for Jetson Orin Nano")
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--width", type=int, default=int(os.getenv("CAMERA_WIDTH", "1920")))
    parser.add_argument("--height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "1080")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("CAMERA_FPS", "30")))
    parser.add_argument("--lang", type=str, default=os.getenv("OCR_LANG", "korean"))
    parser.add_argument("--font-path", type=str, default=os.getenv("OCR_FONT_PATH", ""))
    parser.add_argument("--font-size", type=int, default=int(os.getenv("OCR_FONT_SIZE", "24")))
    parser.add_argument("--det-model-dir", type=str, default=os.getenv("DET_MODEL_DIR", ""))
    parser.add_argument("--rec-model-dir", type=str, default=os.getenv("REC_MODEL_DIR", ""))
    parser.add_argument("--cls-model-dir", type=str, default=os.getenv("CLS_MODEL_DIR", ""))
    parser.add_argument("--use-tensorrt", action="store_true", default=os.getenv("USE_TENSORRT", "0") == "1")
    args = parser.parse_args()

    return OCRRuntimeConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        lang=args.lang,
        use_tensorrt=args.use_tensorrt,
        det_model_dir=args.det_model_dir or None,
        rec_model_dir=args.rec_model_dir or None,
        cls_model_dir=args.cls_model_dir or None,
        font_path=args.font_path or None,
        font_size=args.font_size,
    )

def resolve_font(font_path: str | None) -> str:
    if font_path and Path(font_path).is_file():
        return font_path

    for candidate in KOREAN_FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate

    raise RuntimeError(
        "No Korean-capable font found. Install fonts-noto-cjk or fonts-nanum, or set OCR_FONT_PATH to a valid font file."
    )

def create_ocr_engine(config: OCRRuntimeConfig) -> object:
    # --- [수정된 부분 시작] ---
    # 1. PaddleOCR 내부에 인자로 넣으면 에러가 나므로, 로깅 레벨을 파이썬 시스템 레벨에서 끕니다.
    logging.getLogger("ppocr").setLevel(logging.ERROR)

    # 2. 장치(GPU/CPU) 설정을 Paddle 엔진 레벨에서 강제로 할당합니다.
    if config.use_gpu:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    # --- [수정된 부분 끝] ---

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit(
            "PaddleOCR is not installed. Run setup_jetson_ocr.sh and install the Jetson paddlepaddle-gpu wheel first."
        ) from exc

    # 3. ocr_kwargs에서 에러를 유발하던 'use_gpu'와 'show_log'를 제거했습니다.
    ocr_kwargs = {
        "lang": config.lang,
        "use_textline_orientation": config.use_textline_orientation,
        "enable_mkldnn": config.enable_mkldnn,
        "ir_optim": config.ir_optim,
        "ocr_version": "PP-OCRv4",
    }

    if config.use_tensorrt:
        ocr_kwargs["use_tensorrt"] = True

    if config.det_model_dir:
        ocr_kwargs["det_model_dir"] = config.det_model_dir
    if config.rec_model_dir:
        ocr_kwargs["rec_model_dir"] = config.rec_model_dir
    if config.cls_model_dir:
        ocr_kwargs["cls_model_dir"] = config.cls_model_dir

    return PaddleOCR(**ocr_kwargs)

def open_camera(config: OCRRuntimeConfig) -> cv2.VideoCapture:
    # 젯슨 환경에서는 V4L2 백엔드를 명시하는 것이 안정적입니다.
    capture = cv2.VideoCapture(config.camera_index, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    sharpen_kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )
    return cv2.filter2D(enhanced, -1, sharpen_kernel)

def normalize_ocr_result(raw_result: Sequence) -> List:
    if not raw_result:
        return []

    if len(raw_result) == 1 and isinstance(raw_result[0], list):
        first_item = raw_result[0]
        if first_item and isinstance(first_item[0], (list, tuple)) and len(first_item[0]) == 2:
            return list(first_item)

    return list(raw_result)

def draw_text_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int] = (255, 255, 255),
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    x, y = origin
    bbox = draw.textbbox((x, y), text, font=font)
    padding_x = 6
    padding_y = 4
    background_box = (
        bbox[0] - padding_x,
        bbox[1] - padding_y,
        bbox[2] + padding_x,
        bbox[3] + padding_y,
    )
    draw.rectangle(background_box, fill=background)
    draw.text((x, y), text, font=font, fill=fill)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def overlay_results(
    frame: np.ndarray,
    detections: Iterable,
    font: ImageFont.FreeTypeFont,
) -> np.ndarray:
    annotated = frame.copy()

    for box, (text, score) in detections:
        polygon = np.array(box, dtype=np.int32)
        cv2.polylines(annotated, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        x_min = int(np.min(polygon[:, 0]))
        y_min = int(np.min(polygon[:, 1]))
        label = f"{text} ({score:.2f})"

        label_y = max(0, y_min - 32)
        annotated = draw_text_label(annotated, label, (x_min, label_y), font)

    return annotated

def build_status_panel(frame: np.ndarray, inference_ms: float, fps: float) -> np.ndarray:
    panel = frame.copy()
    status_lines = [
        f"Inference Latency (ms): {inference_ms:.1f}",
        f"FPS: {fps:.1f}",
    ]

    y = 20
    for line in status_lines:
        cv2.rectangle(panel, (15, y - 18), (430, y + 18), (0, 0, 0), thickness=-1)
        cv2.putText(panel, line, (25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 42

    return panel

def run() -> None:
    config = parse_args()
    font_path = resolve_font(config.font_path)
    font = ImageFont.truetype(font_path, config.font_size)

    print("Initializing PaddleOCR...")
    ocr = create_ocr_engine(config)

    capture = open_camera(config)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {config.camera_index}. Check the camera connection and permissions.")

    # SSH 등 화면이 없는 환경에서는 이 부분이 에러를 낼 수 있으므로 주의가 필요합니다.
    cv2.namedWindow("Jetson OCR", cv2.WINDOW_NORMAL)

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                print("Failed to read a frame from the camera.")
                break

            enhanced = enhance_frame(frame)

            infer_start = time.perf_counter()
            raw_result = ocr.predict(enhanced, cls=True)
            inference_ms = (time.perf_counter() - infer_start) * 1000.0

            detections = normalize_ocr_result(raw_result)
            
            # 터미널에도 결과 출력 (디버깅용)
            for _, (text, score) in detections:
                print(f"📝 {text} ({score:.2f})")

            annotated = overlay_results(enhanced, detections, font)

            total_latency_ms = (time.perf_counter() - loop_start) * 1000.0
            fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

            output = build_status_panel(annotated, inference_ms, fps)

            cv2.imshow("Jetson OCR", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()