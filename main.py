import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import base64
import io
import cv2
import datetime
import numpy as np
import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Security
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from paddleocr import PaddleOCR

# Load .env environment variables
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "alxicorn_pp_ocr_secure_secret_key_2026_x89f")
TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "720"))
raw_emails = os.getenv("ALLOWED_EMAILS", "admin@alxicorn.com,dev@alxicorn.com,user@example.com")
ALLOWED_EMAILS = {e.strip().lower() for e in raw_emails.split(",") if e.strip()}

tags_metadata = [
    {
        "name": "Authentication",
        "description": "Token generation service for authorized emails.",
    },
    {
        "name": "OCR Pipeline",
        "description": "Protected end-to-end OCR processing (Detection + Recognition in 1 request).",
    },
    {
        "name": "Detection",
        "description": "Protected text bounding box detection service.",
    },
    {
        "name": "Recognition",
        "description": "Protected batched text recognition service.",
    },
    {
        "name": "Health & UI",
        "description": "Service status and Web UI endpoints.",
    },
]

app = FastAPI(
    title="PP-OCRv6 Microservice API",
    description="""
    ## PP-OCRv6 High-Performance Vision Microservice with Token Authentication
    Secured text detection & text recognition APIs powered by **PaddleOCR**.
    
    * **Web UI Dashboard**: `/`
    * **Swagger Interactive Docs**: `/docs`
    * **ReDoc Interactive Docs**: `/redoc`
    """,
    version="1.1.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Dynamic Device Detection (GPU if available, else CPU)
import paddle

has_gpu = False
try:
    has_gpu = paddle.is_compiled_with_cuda() and ("gpu" in paddle.get_device().lower())
except Exception:
    has_gpu = False

DEVICE = "gpu" if has_gpu else "cpu"

DET_MODEL = "PP-OCRv6_medium_det" if DEVICE == "gpu" else "PP-OCRv4_mobile_det"
REC_MODEL = "PP-OCRv6_medium_rec" if DEVICE == "gpu" else "PP-OCRv4_mobile_rec"

print(f"--> PP-OCR initializing on target device: '{DEVICE.upper()}' using [{DET_MODEL} + {REC_MODEL}]")

det_engine = PaddleOCR(
    text_detection_model_name=DET_MODEL,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device=DEVICE,
    enable_mkldnn=False,
)

rec_engine = PaddleOCR(
    text_recognition_model_name=REC_MODEL,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device=DEVICE,
    enable_mkldnn=False,
)

# ---------------------------------------------------------------------------
# Authentication & Security Helpers
# ---------------------------------------------------------------------------
security_bearer = HTTPBearer(auto_error=False)

def verify_api_token(
    credentials: HTTPAuthorizationCredentials = Security(security_bearer),
    x_api_token: str | None = Header(default=None, alias="X-API-Token")
):
    token_str = None
    if credentials and credentials.credentials:
        token_str = credentials.credentials
    elif x_api_token:
        token_str = x_api_token

    if not token_str:
        raise HTTPException(
            status_code=401,
            detail="Authentication token required. Obtain a token via POST /auth/token or the Web UI."
        )

    try:
        payload = jwt.decode(token_str, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please generate a new token.")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    email: str


class RecBatchRequest(BaseModel):
    images: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "images": [
                        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                    ]
                }
            ]
        }
    }


def _decode_image_bytes(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------------------
# Authentication Endpoint
# ---------------------------------------------------------------------------
@app.post("/auth/token", tags=["Authentication"], summary="Generate API Token by Email")
def generate_token(payload: TokenRequest):
    """
    Generates a secure API token if the submitted email exists in the ALLOWED_EMAILS list in .env.
    """
    email_clean = payload.email.strip().lower()
    if ALLOWED_EMAILS and email_clean not in ALLOWED_EMAILS:
        raise HTTPException(
            status_code=401,
            detail=f"Email '{payload.email}' is not authorized to generate API tokens."
        )

    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=TOKEN_EXPIRE_HOURS)
    access_token = jwt.encode({"sub": email_clean, "exp": expire}, SECRET_KEY, algorithm="HS256")

    return {
        "status": "success",
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in_hours": TOKEN_EXPIRE_HOURS,
        "email": email_clean
    }


# ---------------------------------------------------------------------------
# Protected OCR Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict/det", tags=["Detection"], summary="Detect Text Bounding Boxes [Protected]")
def detect_text_boxes(file: UploadFile = File(...), token_payload: dict = Depends(verify_api_token)):
    contents = file.file.read()
    img = _decode_image_bytes(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    result = det_engine.predict(img)
    boxes = []

    for res in result:
        rec_boxes = res.get("rec_boxes", [])
        for idx, box in enumerate(rec_boxes):
            x_min, y_min, x_max, y_max = [int(v) for v in box.tolist()]
            boxes.append({
                "box_id": idx,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": x_max - x_min,
                "height": y_max - y_min
            })

    return {
        "status": "success",
        "boxes_count": len(boxes),
        "boxes": boxes
    }


@app.post("/predict/rec", tags=["Recognition"], summary="Recognize Text Batch from Base64 Crops [Protected]")
def recognize_text_batch(payload: RecBatchRequest, token_payload: dict = Depends(verify_api_token)):
    if not payload.images:
        return {"status": "success", "results": []}

    cropped_imgs = []
    for b64_str in payload.images:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            cropped_imgs.append(img)

    if not cropped_imgs:
        raise HTTPException(status_code=400, detail="No valid images to process")

    predictions = rec_engine.predict(cropped_imgs)
    results = []

    for idx, pred in enumerate(predictions):
        texts = pred.get("rec_texts", [""])
        scores = pred.get("rec_scores", [0.0])
        text = texts[0].strip() if texts else ""
        score = float(scores[0]) if scores else 0.0

        results.append({
            "index": idx,
            "text": text,
            "score": score
        })

    return {
        "status": "success",
        "results": results
    }


@app.post("/predict/ocr", tags=["OCR Pipeline"], summary="Full End-to-End OCR Pipeline [Protected]")
def full_pipeline_ocr(
    file: UploadFile = File(...),
    min_score: float = 0.90,
    det_max_side: int = 960,
    token_payload: dict = Depends(verify_api_token)
):
    contents = file.file.read()
    img = _decode_image_bytes(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    h, w = img.shape[:2]

    if det_max_side > 0 and max(h, w) > det_max_side:
        scale = det_max_side / float(max(h, w))
        det_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        det_img = img

    det_result = det_engine.predict(det_img)
    valid_crops = []
    box_metadata = []

    for res in det_result:
        rec_boxes = res.get("rec_boxes", [])
        for idx, box in enumerate(rec_boxes):
            x_min = int(round(box[0] / scale))
            y_min = int(round(box[1] / scale))
            x_max = int(round(box[2] / scale))
            y_max = int(round(box[3] / scale))

            x_min = max(0, min(x_min, w - 1))
            y_min = max(0, min(y_min, h - 1))
            x_max = max(0, min(x_max, w))
            y_max = max(0, min(y_max, h))

            crop_w = x_max - x_min
            crop_h = y_max - y_min

            if crop_w < 5 or crop_h < 5:
                continue

            crop = img[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                valid_crops.append(crop)
                box_metadata.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": crop_w,
                    "height": crop_h
                })

    if not valid_crops:
        return {"status": "success", "words_count": 0, "lines": []}

    rec_predictions = rec_engine.predict(valid_crops)

    words = []
    for idx, (box, pred) in enumerate(zip(box_metadata, rec_predictions)):
        texts = pred.get("rec_texts", [""])
        scores = pred.get("rec_scores", [0.0])
        text = texts[0].strip() if texts else ""
        score = float(scores[0]) if scores else 0.0

        if not text or score < min_score:
            continue

        words.append({
            "WordText": text,
            "Left": box["x_min"],
            "Top": box["y_min"],
            "Width": box["width"],
            "Height": box["height"],
            "Score": score
        })

    lines = [{"LineText": w["WordText"], "Words": [w]} for w in words]

    return {
        "status": "success",
        "words_count": len(words),
        "lines": lines
    }


# ---------------------------------------------------------------------------
# Single-Page Web UI Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Health & UI"], summary="Web Dashboard UI")
def serve_web_ui():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PP-OCRv6 Vision Microservice</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #06b6d4;
            --text-light: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --error: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }

        body {
            background: radial-gradient(circle at top left, #1e1b4b, #0f172a 50%, #020617);
            color: var(--text-light);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(12px);
            background: rgba(15, 23, 42, 0.6);
        }

        .logo { font-weight: 700; font-size: 1.3rem; letter-spacing: -0.5px; display: flex; align-items: center; gap: 0.75rem; }
        .logo-badge { background: linear-gradient(135deg, var(--primary), var(--accent)); padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; text-transform: uppercase; }

        .nav-links a { color: var(--text-muted); text-decoration: none; font-size: 0.9rem; transition: color 0.2s; margin-left: 1.5rem; }
        .nav-links a:hover { color: var(--text-light); }

        container { max-width: 1100px; margin: 2.5rem auto; padding: 0 1.5rem; width: 100%; flex: 1; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        @media (max-width: 850px) { .grid { grid-template-columns: 1fr; } }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
        }

        .card-header { margin-bottom: 1.5rem; }
        .card-title { font-size: 1.25rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; }
        .card-desc { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.3rem; }

        .form-group { margin-bottom: 1.25rem; }
        label { display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--text-muted); }
        input[type="email"], input[type="text"] {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            padding: 0.8rem 1rem;
            border-radius: 10px;
            color: var(--text-light);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus { border-color: var(--primary); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }

        .btn {
            background: linear-gradient(135deg, var(--primary), var(--primary-hover));
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            font-size: 0.95rem;
            transition: transform 0.1s, opacity 0.2s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }
        .btn:hover { opacity: 0.95; }
        .btn:active { transform: scale(0.98); }

        .drop-zone {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 2rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
            background: rgba(15, 23, 42, 0.4);
            margin-bottom: 1.25rem;
        }
        .drop-zone:hover, .drop-zone.dragover { border-color: var(--accent); background: rgba(6, 182, 212, 0.05); }

        .preview-img { max-width: 100%; max-height: 200px; border-radius: 8px; margin-top: 1rem; display: none; }

        .token-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
            word-break: break-all;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--accent);
            position: relative;
        }

        .copy-btn {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-light);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
        }
        .copy-btn:hover { background: var(--primary); }

        .badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 6px; font-size: 0.75rem; font-weight: 500; }
        .badge-success { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .badge-error { background: rgba(239, 68, 68, 0.2); color: var(--error); }

        .results-box {
            margin-top: 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 1rem;
            max-height: 250px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }

        .ocr-line-item {
            padding: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
        }

        footer { text-align: center; padding: 1.5rem; color: var(--text-muted); font-size: 0.8rem; border-top: 1px solid var(--border-color); }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span>⚡ PP-OCRv6</span>
            <span class="logo-badge">Microservice</span>
        </div>
        <div class="nav-links">
            <a href="/docs" target="_blank">Swagger Docs</a>
            <a href="/redoc" target="_blank">ReDoc</a>
        </div>
    </header>

    <container>
        <div class="grid">
            <!-- TOKEN GENERATOR CARD -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🔐 Generate API Access Token</div>
                    <div class="card-desc">Enter your authorized email to generate a signed JWT bearer token.</div>
                </div>

                <div class="form-group">
                    <label for="emailInput">Authorized Email</label>
                    <input type="email" id="emailInput" placeholder="Enter authorized email address">
                </div>

                <button class="btn" onclick="generateToken()">
                    <span>Generate Token</span>
                </button>

                <div id="tokenResultArea" style="display: none;">
                    <div style="margin-top: 1rem; display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge badge-success">✓ Authorized</span>
                        <span style="font-size: 0.75rem; color: var(--text-muted);">Saved to Browser Storage</span>
                    </div>
                    <div class="token-box">
                        <button class="copy-btn" onclick="copyToken()">Copy</button>
                        <span id="tokenText"></span>
                    </div>
                </div>

                <div id="authErrorArea" style="display: none; margin-top: 1rem;">
                    <span class="badge badge-error" id="authErrorMsg"></span>
                </div>
            </div>

            <!-- LIVE OCR TESTER CARD -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🔍 Live OCR Tester</div>
                    <div class="card-desc">Upload a screenshot image to execute end-to-end OCR processing.</div>
                </div>

                <div class="form-group">
                    <label for="tokenInput">Access Token (Bearer)</label>
                    <input type="text" id="tokenInput" placeholder="Paste or generate access token above">
                </div>

                <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📷</div>
                    <div style="font-size: 0.9rem; color: var(--text-light);">Drag & Drop screenshot here</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.3rem;">or click to browse (.png, .jpg)</div>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
                    <img id="previewImg" class="preview-img" alt="Preview">
                </div>

                <button class="btn" id="ocrBtn" onclick="runOCR()">
                    <span>Run OCR Pipeline</span>
                </button>

                <div id="ocrStatus" style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted); text-align: center;"></div>

                <div id="ocrResults" class="results-box" style="display: none;"></div>
            </div>
        </div>
    </container>

    <footer>
        PP-OCRv6 Vision Microservice &copy; 2026 Alxicorn. All rights reserved.
    </footer>

    <script>
        let selectedFile = null;

        // Auto load saved token
        window.addEventListener('DOMContentLoaded', () => {
            const savedToken = localStorage.getItem('ocr_api_token');
            if (savedToken) {
                document.getElementById('tokenInput').value = savedToken;
            }
        });

        async function generateToken() {
            const email = document.getElementById('emailInput').value.trim();
            const tokenArea = document.getElementById('tokenResultArea');
            const errorArea = document.getElementById('authErrorArea');
            const errorMsg = document.getElementById('authErrorMsg');

            tokenArea.style.display = 'none';
            errorArea.style.display = 'none';

            if (!email) {
                errorMsg.innerText = 'Please enter a valid email address.';
                errorArea.style.display = 'block';
                return;
            }

            try {
                const res = await fetch('/auth/token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email: email })
                });

                const data = await res.json();
                if (res.ok && data.access_token) {
                    document.getElementById('tokenText').innerText = data.access_token;
                    document.getElementById('tokenInput').value = data.access_token;
                    localStorage.setItem('ocr_api_token', data.access_token);
                    tokenArea.style.display = 'block';
                } else {
                    errorMsg.innerText = data.detail || 'Email unauthorized';
                    errorArea.style.display = 'block';
                }
            } catch (err) {
                errorMsg.innerText = 'Error connecting to auth endpoint';
                errorArea.style.display = 'block';
            }
        }

        function copyToken() {
            const token = document.getElementById('tokenText').innerText;
            navigator.clipboard.writeText(token);
            alert('API Token copied to clipboard!');
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                const preview = document.getElementById('previewImg');
                preview.src = URL.createObjectURL(file);
                preview.style.display = 'inline-block';
            }
        }

        // Drag and drop support
        const dropZone = document.getElementById('dropZone');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                selectedFile = e.dataTransfer.files[0];
                const preview = document.getElementById('previewImg');
                preview.src = URL.createObjectURL(selectedFile);
                preview.style.display = 'inline-block';
            }
        });

        async function runOCR() {
            const token = document.getElementById('tokenInput').value.trim();
            const statusDiv = document.getElementById('ocrStatus');
            const resultsDiv = document.getElementById('ocrResults');

            resultsDiv.style.display = 'none';
            resultsDiv.innerHTML = '';

            if (!token) {
                statusDiv.innerText = '⚠️ Please enter or generate an Access Token first.';
                statusDiv.style.color = 'var(--error)';
                return;
            }

            if (!selectedFile) {
                statusDiv.innerText = '⚠️ Please select or drop an image file.';
                statusDiv.style.color = 'var(--error)';
                return;
            }

            statusDiv.innerText = '⏳ Processing image OCR pipeline...';
            statusDiv.style.color = 'var(--accent)';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const res = await fetch('/predict/ocr', {
                    method: 'POST',
                    headers: { 'Authorization': 'Bearer ' + token },
                    body: formData
                });

                const data = await res.json();
                if (res.ok) {
                    statusDiv.innerText = `✓ Successfully extracted ${data.words_count || 0} words in ${data.lines ? data.lines.length : 0} lines!`;
                    statusDiv.style.color = 'var(--success)';

                    let html = '';
                    if (data.lines && data.lines.length) {
                        data.lines.forEach(line => {
                            const score = line.Words && line.Words[0] ? (line.Words[0].Score * 100).toFixed(1) + '%' : '';
                            html += `<div class="ocr-line-item"><span>${line.LineText}</span><span style="color: var(--accent);">${score}</span></div>`;
                        });
                    } else {
                        html = '<div style="padding: 0.5rem; color: var(--text-muted);">No text detected in image.</div>';
                    }
                    resultsDiv.innerHTML = html;
                    resultsDiv.style.display = 'block';
                } else {
                    statusDiv.innerText = '❌ Error: ' + (data.detail || 'Failed to process OCR');
                    statusDiv.style.color = 'var(--error)';
                }
            } catch (err) {
                statusDiv.innerText = '❌ Network error during OCR request.';
                statusDiv.style.color = 'var(--error)';
            }
        }
    </script>
</body>
</html>
    """
    return html_content
