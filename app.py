"""
Jetson Orin Nano Real-time OCR Web Streaming Server
Using PaddleOCR (PP-OCRv3) with Flask for browser-based visualization
"""
import cv2
import paddle
import logging
import numpy as np
from flask import Flask, Response, render_template_string
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# 1. Flask 앱 초기화
app = Flask(__name__)

# 2. 패들 장치 및 로깅 설정
logging.getLogger("ppocr").setLevel(logging.ERROR)
paddle.set_device('gpu')

# 3. OCR 엔진 초기화
print("⏳ PaddleOCR 초기화 중...")
ocr = PaddleOCR(lang='korean', ocr_version='PP-OCRv3', use_angle_cls=False, show_log=False)
print("✅ PaddleOCR 로드 완료!")

# 4. 한글 폰트 설정 (젯슨 우분투 기본 나눔폰트 경로)
# 폰트가 없다면 sudo apt-get install fonts-nanum 으로 설치
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
try:
    font = ImageFont.truetype(font_path, 20)
except IOError:
    font = ImageFont.load_default()
    print("⚠️ 경고: 나눔폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

def draw_ocr_results(frame, results):
    """프레임에 바운딩 박스와 텍스트를 그려 반환합니다."""
    if not results or not results[0]:
        return frame

    # 한글 출력을 위해 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for line in results[0]:
        if isinstance(line, list) and len(line) == 2:
            box = line[0]
            content = line[1]

            if isinstance(content, (tuple, list)) and len(content) == 2:
                text = content[0]
                score = content[1]

                if score > 0.6:
                    # OCR은 절반 크기(320x240)에서 수행했으므로, 좌표를 2배로 늘림(640x480)
                    scaled_box = [(int(p[0] * 2), int(p[1] * 2)) for p in box]
                    
                    # 바운딩 박스 그리기
                    draw.polygon(scaled_box, outline=(0, 255, 0), width=2)
                    
                    # 텍스트 그리기 (박스 좌상단 기준)
                    text_position = (scaled_box[0][0], max(0, scaled_box[0][1] - 25))
                    draw.text(text_position, f"{text} ({score:.2f})", font=font, fill=(0, 255, 0))

    # 다시 OpenCV 형식으로 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_frames():
    """웹캠 프레임을 지속적으로 읽고 OCR을 적용하여 JPEG로 변환하는 제너레이터"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 속도 향상을 위해 OCR 추론은 절반 크기에서 진행
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        result = ocr.ocr(small_frame)
        
        # 원본 프레임에 결과 그리기
        annotated_frame = draw_ocr_results(frame, result)

        # 웹 스트리밍을 위해 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # MJPEG 스트리밍 포맷으로 yield
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """브라우저 접속 시 보여줄 심플한 HTML 페이지"""
    html = """
    <html>
    <head>
        <title>Jetson OCR Live Stream</title>
        <style>
            body { 
                background-color: #222; 
                color: white; 
                text-align: center; 
                font-family: sans-serif;
                padding: 20px;
            }
            h2 { margin: 10px 0; }
            .container {
                max-width: 800px;
                margin: 20px auto;
                background-color: #333;
                padding: 20px;
                border-radius: 10px;
            }
            img { 
                border: 3px solid #444; 
                border-radius: 10px; 
                margin-top: 20px; 
                max-width: 100%;
                box-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
            }
            .info { 
                font-size: 0.9em; 
                color: #aaa; 
                margin-top: 15px; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚀 Jetson Orin Nano - Live OCR</h2>
            <p>Real-time Korean/English text recognition using PaddleOCR (PP-OCRv3)</p>
            <img src="/video_feed" width="640" height="480" />
            <div class="info">
                <p>🟢 Live streaming • GPU accelerated • Compute Capability 8.7</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍 라우트"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 외부(노트북)에서 접속할 수 있도록 0.0.0.0으로 호스팅
    print("\n" + "="*60)
    print("🌐 Flask Server Starting...")
    print("📍 Access from browser: http://<jetson-ip>:5000")
    print("💡 For local: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
