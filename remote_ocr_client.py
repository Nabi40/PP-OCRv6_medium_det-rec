import base64
import cv2
import requests
import numpy as np

SERVER_URL = "http://localhost:8010"
AUTH_API_URL = f"{SERVER_URL}/auth/token"
DET_API_URL = f"{SERVER_URL}/predict/det"
REC_API_URL = f"{SERVER_URL}/predict/rec"
OCR_API_URL = f"{SERVER_URL}/predict/ocr"

# Cache token in memory to avoid requesting token on every OCR call
_CACHED_TOKEN = None

def get_access_token(email: str = "alxicornteam.akmahamudunnabi@gmail.com") -> str:
    """
    Fetches an access token from the auth endpoint using an authorized email.
    """
    global _CACHED_TOKEN
    if _CACHED_TOKEN:
        return _CACHED_TOKEN

    response = requests.post(AUTH_API_URL, json={"email": email})
    if response.status_code == 200:
        _CACHED_TOKEN = response.json().get("access_token")
        return _CACHED_TOKEN
    else:
        raise PermissionError(f"Failed to get token for email '{email}': {response.text}")


def remote_ocr_lines(
    bgr_image: np.ndarray,
    min_score: float = 0.90,
    email: str = "alxicornteam.akmahamudunnabi@gmail.com",
    token: str = None,
    use_full_pipeline_endpoint: bool = True
) -> list:
    """
    Queries the remote PP-OCR microservice on port 8000 and returns
    compatible Lines/Words structures. Automatically handles Bearer Token authorization.

    :param bgr_image: Image array in OpenCV BGR format.
    :param min_score: Minimum confidence threshold.
    :param email: Authorized email configured in server's .env.
    :param token: Pre-existing token string (optional).
    :param use_full_pipeline_endpoint: If True, uses single POST /predict/ocr call.
    """
    if not token:
        token = get_access_token(email)

    headers = {"Authorization": f"Bearer {token}"}

    _, img_encoded = cv2.imencode(".png", bgr_image)
    img_bytes = img_encoded.tobytes()

    if use_full_pipeline_endpoint:
        response = requests.post(
            f"{OCR_API_URL}?min_score={min_score}",
            headers=headers,
            files={"file": ("screen.png", img_bytes, "image/png")}
        )
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("lines", [])

    # Step 1: Post screenshot to Detection API
    response_det = requests.post(
        DET_API_URL,
        headers=headers,
        files={"file": ("screen.png", img_bytes, "image/png")}
    )
    if response_det.status_code != 200:
        return []

    det_data = response_det.json()
    boxes = det_data.get("boxes", [])
    if not boxes:
        return []

    # Step 2: Crop image patches & prepare Base64 batch for Recognition API
    cropped_b64_list = []
    valid_boxes = []

    for box in boxes:
        x_min, y_min = box["x_min"], box["y_min"]
        x_max, y_max = box["x_max"], box["y_max"]
        crop = bgr_image[y_min:y_max, x_min:x_max]

        if crop.size == 0:
            continue

        _, crop_buf = cv2.imencode(".png", crop)
        b64_str = base64.b64encode(crop_buf.tobytes()).decode("utf-8")
        cropped_b64_list.append(b64_str)
        valid_boxes.append(box)

    if not cropped_b64_list:
        return []

    # Step 3: Post crop batch to Recognition API
    response_rec = requests.post(REC_API_URL, headers=headers, json={"images": cropped_b64_list})
    if response_rec.status_code != 200:
        return []

    rec_data = response_rec.json()
    rec_results = rec_data.get("results", [])

    # Step 4: Reassemble into WordText dicts
    words = []
    for box, res in zip(valid_boxes, rec_results):
        text = res.get("text", "").strip()
        score = res.get("score", 0.0)

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

    if not words:
        return []

    return [{"LineText": w["WordText"], "Words": [w]} for w in words]
