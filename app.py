"""
Jetson Orin Nano Real-time OCR Web Streaming Server
Using PaddleOCR (PP-OCRv3) with Flask for browser-based visualization
"""
import cv2
import paddle
import logging
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
import json
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import threading

# 1. Flask 앱 초기화
app = Flask(__name__)

# 2. 패들 장치 및 로깅 설정
logging.getLogger("ppocr").setLevel(logging.ERROR)
paddle.set_device('gpu')

# 3. 전역 변수: 실시간 OCR 결과 저장
ocr_results_lock = threading.Lock()
ocr_results = {"texts": [], "timestamp": 0, "frame_count": 0, "detected_count": 0, "average_score": 0.0}

# 3. OCR 엔진 초기화
print("⏳ PaddleOCR 초기화 중...")
ocr = PaddleOCR(lang='korean', ocr_version='PP-OCRv3', use_angle_cls=False)
print("✅ PaddleOCR 로드 완료!")

# 4. 한글 폰트 설정 (젯슨 우분투 기본 나눔폰트 경로)
# 폰트가 없다면 sudo apt-get install fonts-nanum 으로 설치
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
try:
    font = ImageFont.truetype(font_path, 20)
except IOError:
    font = ImageFont.load_default()
    print("⚠️ 경고: 나눔폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

def extract_ocr_text(results):
    """OCR 결과에서 텍스트만 추출합니다."""
    text_list = []
    if not results:
        return text_list

    prediction = results[0] if isinstance(results, list) and results else results

    if isinstance(prediction, dict) and "rec_texts" in prediction:
        rec_texts = prediction.get("rec_texts", [])
        rec_scores = prediction.get("rec_scores", [])
        for text, score in zip(rec_texts, rec_scores):
            if score > 0.4:
                text_list.append({"text": text, "score": f"{score:.2f}"})
        text_list.sort(key=lambda item: float(item["score"]), reverse=True)
        return text_list[:10]

    for line in prediction or []:
        if isinstance(line, list) and len(line) == 2:
            content = line[1]
            if isinstance(content, (tuple, list)) and len(content) == 2:
                text = content[0]
                score = content[1]
                if score > 0.4:  # 커트라인을 40%로 대폭 낮춥니다.
                    text_list.append({"text": text, "score": f"{score:.2f}"})
    text_list.sort(key=lambda item: float(item["score"]), reverse=True)
    return text_list[:10]

def summarize_ocr_text(text_list):
    """OCR 텍스트 목록의 요약 정보를 계산합니다."""
    if not text_list:
        return 0, 0.0

    scores = [float(item["score"]) for item in text_list]
    return len(text_list), round(sum(scores) / len(scores), 2)

def draw_ocr_results(frame, results):
    """프레임에 바운딩 박스와 텍스트를 그려 반환합니다."""
    if not results:
        return frame

    prediction = results[0] if isinstance(results, list) and results else results

    # 한글 출력을 위해 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    if isinstance(prediction, dict) and "rec_texts" in prediction:
        rec_texts = prediction.get("rec_texts", [])
        rec_scores = prediction.get("rec_scores", [])
        rec_polys = prediction.get("rec_polys", [])

        for text, score, box in zip(rec_texts, rec_scores, rec_polys):
            if score > 0.4 and box is not None:
                scaled_box = [(int(p[0]), int(p[1])) for p in box]
                draw.polygon(scaled_box, outline=(0, 255, 0), width=2)
                text_position = (scaled_box[0][0], max(0, scaled_box[0][1] - 25))
                draw.text(text_position, f"{text} ({score:.2f})", font=font, fill=(0, 255, 0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    for line in prediction or []:
        if isinstance(line, list) and len(line) == 2:
            box = line[0]
            content = line[1]

            if isinstance(content, (tuple, list)) and len(content) == 2:
                text = content[0]
                score = content[1]

                if score > 0.4:
                    # 프레임을 줄이지 않으므로 좌표를 그대로 사용합니다.
                    scaled_box = [(int(p[0]), int(p[1])) for p in box]
                    
                    # 바운딩 박스 그리기
                    draw.polygon(scaled_box, outline=(0, 255, 0), width=2)
                    
                    # 텍스트 그리기 (박스 좌상단 기준)
                    text_position = (scaled_box[0][0], max(0, scaled_box[0][1] - 25))
                    draw.text(text_position, f"{text} ({score:.2f})", font=font, fill=(0, 255, 0))

    # 다시 OpenCV 형식으로 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def generate_frames():
    """웹캠 프레임을 지속적으로 읽고 OCR을 적용하여 JPEG로 변환하는 제너레이터"""
    global ocr_results
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # 원본(640x480) 프레임을 그대로 OCR에 넣습니다.
        result = ocr.ocr(frame)

        # AI가 실제로 보고 있는 OCR 원시 데이터를 주기적으로 출력합니다.
        if frame_count % 10 == 0:
            print(f"🔍 [프레임 {frame_count}] OCR 데이터: {result}")
        
        # OCR 결과를 텍스트만 추출하여 전역 변수에 저장
        text_list = extract_ocr_text(result)
        detected_count, average_score = summarize_ocr_text(text_list)
        with ocr_results_lock:
            ocr_results["texts"] = text_list
            ocr_results["frame_count"] = frame_count
            ocr_results["detected_count"] = detected_count
            ocr_results["average_score"] = average_score
        
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
    """브라우저 접속 시 보여줄 HTML 페이지 (비디오 + 텍스트 결과 2컬럼)"""
    html = """
    <html>
    <head>
        <title>Jetson OCR Live Stream</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                background-color: #1a1a1a; 
                color: white; 
                font-family: 'Segoe UI', sans-serif;
                padding: 20px;
                min-height: 100vh;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            h1 { 
                font-size: 2em;
                margin-bottom: 5px;
                color: #00ff00;
            }
            .subtitle { 
                color: #aaa; 
                font-size: 0.9em;
            }
            .main-container {
                display: flex;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
                flex-wrap: wrap;
            }
            .video-panel {
                flex: 1;
                min-width: 400px;
                background-color: #222;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
            }
            .video-panel h3 {
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            img { 
                display: block;
                width: 100%;
                border: 2px solid #00ff00;
                border-radius: 8px;
                max-width: 640px;
            }
            .results-panel {
                flex: 1;
                min-width: 300px;
                background-color: #222;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
                display: flex;
                flex-direction: column;
            }
            .results-panel h3 {
                margin-bottom: 15px;
                font-size: 1.1em;
                border-bottom: 2px solid #00ff00;
                padding-bottom: 10px;
            }
            #results-list {
                flex: 1;
                overflow-y: auto;
                list-style: none;
            }
            .result-item {
                background-color: #333;
                padding: 10px;
                margin-bottom: 8px;
                border-radius: 5px;
                border-left: 3px solid #00ff00;
                animation: slideIn 0.3s ease-in;
            }
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateX(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            .result-text {
                font-weight: bold;
                color: #00ff00;
                font-size: 1.1em;
                margin-bottom: 5px;
            }
            .result-score {
                font-size: 0.85em;
                color: #888;
            }
            .info { 
                color: #666; 
                font-size: 0.85em; 
                margin-top: 15px; 
                text-align: center;
            }
            .status {
                font-size: 0.9em;
                color: #00dd00;
                margin-top: 10px;
            }
            .summary {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                margin-bottom: 15px;
            }
            .summary-card {
                background: #333;
                border-radius: 8px;
                padding: 12px;
                border: 1px solid rgba(0, 255, 0, 0.18);
            }
            .summary-label {
                color: #aaa;
                font-size: 0.8em;
                margin-bottom: 6px;
                letter-spacing: 0.02em;
            }
            .summary-value {
                color: #00ff00;
                font-size: 1.2em;
                font-weight: 700;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Jetson Orin Nano - Live OCR</h1>
            <p class="subtitle">Real-time Korean/English text recognition using PaddleOCR (PP-OCRv3)</p>
        </div>

        <div class="main-container">
            <div class="video-panel">
                <h3>📹 Video Stream</h3>
                <img src="/video_feed" alt="Live Video Feed" />
                <div class="info">🟢 Live streaming • GPU accelerated • Compute 8.7</div>
            </div>

            <div class="results-panel">
                <h3>📄 Recognized Text</h3>
                <div class="summary">
                    <div class="summary-card">
                        <div class="summary-label">표시 중인 텍스트 수</div>
                        <div class="summary-value" id="detected-count">0</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">평균 신뢰도</div>
                        <div class="summary-value" id="average-score">0.00</div>
                    </div>
                </div>
                <ul id="results-list">
                    <li class="result-item" style="color: #888;">대기 중...</li>
                </ul>
                <div class="status">Frame: <span id="frame-count">0</span></div>
                <div class="status">Top 10 by confidence</div>
            </div>
        </div>

        <script>
            // 실시간으로 OCR 결과 업데이트
            function updateResults() {
                fetch('/ocr_results')
                    .then(response => response.json())
                    .then(data => {
                        const resultsList = document.getElementById('results-list');
                        const frameCount = document.getElementById('frame-count');
                        const detectedCount = document.getElementById('detected-count');
                        const averageScore = document.getElementById('average-score');
                        
                        frameCount.textContent = data.frame_count;
                        detectedCount.textContent = data.detected_count ?? 0;
                        averageScore.textContent = Number(data.average_score ?? 0).toFixed(2);
                        
                        if (data.texts && data.texts.length > 0) {
                            resultsList.innerHTML = '';
                            data.texts.forEach(item => {
                                const li = document.createElement('li');
                                li.className = 'result-item';
                                li.innerHTML = `
                                    <div class="result-text">${escapeHtml(item.text)}</div>
                                    <div class="result-score">신뢰도: ${item.score}</div>
                                `;
                                resultsList.appendChild(li);
                            });
                        } else {
                            resultsList.innerHTML = '<li class="result-item" style="color: #888;">인식된 텍스트 없음</li>';
                        }
                    })
                    .catch(err => console.error('Error updating results:', err));
            }

            // HTML 이스케이프 함수 (XSS 방지)
            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            // 初始 업데이트 및 주기적 폴링 (500ms)
            updateResults();
            setInterval(updateResults, 500);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    """비디오 스트리밍 라우트"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ocr_results')
def get_ocr_results():
    """실시간 OCR 결과를 JSON으로 반환"""
    global ocr_results
    with ocr_results_lock:
        return jsonify(ocr_results)

if __name__ == '__main__':
    # 외부(노트북)에서 접속할 수 있도록 0.0.0.0으로 호스팅
    print("\n" + "="*60)
    print("🌐 Flask Server Starting...")
    print("📍 Access from browser: http://<jetson-ip>:5000")
    print("💡 For local: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
