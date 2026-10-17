from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from face_engine import (
    DETECTION_MODEL,
    MATCH_THRESHOLD,
    RECOGNITION_MODEL,
    RECOGNITION_MODEL_ID,
    Face,
    FaceEngine,
    cosine_similarity,
    decode_embedding,
    decode_image,
    encode_embedding,
)


app = FastAPI(title="Faceproof API")

# Validation thresholds. Tune these against your real accepted/rejected photos.
MIN_ASPECT_RATIO = 0.55
MAX_ASPECT_RATIO = 0.90
MIN_FACE_AREA_RATIO = 0.10
MIN_FACE_HEIGHT_RATIO = 0.25
MAX_FACE_HEIGHT_RATIO = 0.70
MAX_FACE_CENTER_OFFSET_X = 0.12
MAX_FACE_CENTER_OFFSET_Y = 0.20
MAX_HEAD_ROLL_DEGREES = 8.0
MAX_NOSE_EYE_OFFSET_RATIO = 0.20
MAX_MOUTH_EYE_OFFSET_RATIO = 0.25
MIN_BACKGROUND_RATIO = 0.80
BACKGROUND_BORDER_RATIO = 0.06
MIN_BLUR_SCORE = 15
MIN_FACE_SCORE = 0.85
# Reference images (ID card, passport page, older photo) are often small or printed.
MIN_REFERENCE_FACE_SCORE = 0.70
# A second face at least this fraction of the main face's area makes the photo ambiguous.
# Smaller ones (e.g. the ghost portrait on an ID card) are ignored for matching.
SECOND_FACE_AREA_RATIO = 0.25

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

ENGINE = FaceEngine()
STATIC_DIR = Path(__file__).resolve().parent / "static"
FACE_DETECTOR_NAME = "YuNet / FaceDetectorYN"
FACE_RECOGNIZER_NAME = "SFace / FaceRecognizerSF"


class BadRequest(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"valid": False, "errors": [message], "metrics": {}}, status_code=status_code)


def build_health_payload() -> tuple[dict[str, Any], int]:
    ready = ENGINE.detector is not None and ENGINE.recognizer is not None

    return (
        {
            "status": "ok" if ready else "degraded",
            "opencv_version": cv2.__version__,
            "face_detector": FACE_DETECTOR_NAME,
            "face_model": DETECTION_MODEL.name,
            "model_loaded": ENGINE.detector is not None,
            "model_error": ENGINE.detector_error,
            "face_recognizer": FACE_RECOGNIZER_NAME,
            "recognition_model": RECOGNITION_MODEL.name,
            "recognition_model_loaded": ENGINE.recognizer is not None,
            "recognition_model_error": ENGINE.recognizer_error,
            "match_threshold": MATCH_THRESHOLD,
        },
        200 if ready else 503,
    )


def add_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def read_upload(upload: UploadFile, label: str) -> bytes:
    if upload.content_type not in ALLOWED_TYPES:
        raise BadRequest(f"Please choose a JPEG or PNG image file for {label}.")

    # Read one byte past the limit instead of buffering an arbitrarily large upload.
    image_bytes = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise BadRequest(f"{label.capitalize()} is too large. Maximum size is 10 MB.")
    return image_bytes


def calculate_head_roll(
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
) -> float:
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    # Depending on landmark ordering, atan2 may produce an angle near +/-180.
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def sideways_offset(
    point: tuple[float, float],
    eye_mid: tuple[float, float],
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    eye_distance: float,
) -> float:
    """Distance of a point from the eye midpoint along the eye line, relative to eye distance."""
    if eye_distance < 1.0:
        return 0.0
    ux = (right_eye[0] - left_eye[0]) / eye_distance
    uy = (right_eye[1] - left_eye[1]) / eye_distance
    return abs((point[0] - eye_mid[0]) * ux + (point[1] - eye_mid[1]) * uy) / eye_distance


def blur_score(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_background_mask(
    image: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """
    Build a conservative background mask from the outer image region.

    We exclude a large rectangle around the detected face/upper body so skin,
    hair and clothing are not incorrectly treated as background pixels.
    """
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8)

    # Ignore the central/lower part of the image where shoulders/body normally are.
    center_x1 = int(width * 0.20)
    center_x2 = int(width * 0.80)
    center_y1 = int(height * 0.18)
    mask[center_y1:height, center_x1:center_x2] = 0

    if face_box is not None:
        x, y, w, h = face_box
        margin_x = int(w * 0.65)
        margin_top = int(h * 0.40)
        margin_bottom = int(h * 1.20)

        # Always keep thin side and top strips: with a normal head-and-shoulders
        # framing the margins would otherwise cover the whole photo and leave
        # no background to check.
        border_x = max(1, int(width * BACKGROUND_BORDER_RATIO))
        border_y = max(1, int(height * BACKGROUND_BORDER_RATIO))

        x1 = max(border_x, x - margin_x)
        y1 = max(border_y, y - margin_top)
        x2 = min(width - border_x, x + w + margin_x)
        y2 = min(height, y + h + margin_bottom)
        mask[y1:y2, x1:x2] = 0

    return mask


def check_background_white_or_blue(
    image: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
) -> tuple[bool, float, str, dict[str, float]]:
    mask = build_background_mask(image, face_box)
    pixels = image[mask == 1]

    if pixels.size == 0:
        return False, 0.0, "unknown", {"white": 0.0, "blue": 0.0}

    # BGR channels as signed integers so subtraction does not wrap at uint8.
    b = pixels[:, 0].astype(np.int16)
    g = pixels[:, 1].astype(np.int16)
    r = pixels[:, 2].astype(np.int16)

    # White: bright and nearly neutral.
    white_mask = (
        (r >= 200)
        & (g >= 200)
        & (b >= 200)
        & (np.abs(r - g) <= 35)
        & (np.abs(r - b) <= 35)
        & (np.abs(g - b) <= 35)
    )

    # Blue: sufficiently saturated blue channel. Supports dark/medium/bright blue.
    blue_mask = (
        (b >= 95)
        & ((b - r) >= 25)
        & ((b - g) >= 15)
    )

    white_ratio = float(np.mean(white_mask))
    blue_ratio = float(np.mean(blue_mask))

    if white_ratio >= blue_ratio:
        background_type = "white"
        dominant_ratio = white_ratio
    else:
        background_type = "blue"
        dominant_ratio = blue_ratio

    # A formal background should mostly be one allowed colour, not a mixture.
    valid = dominant_ratio >= MIN_BACKGROUND_RATIO

    return (
        valid,
        dominant_ratio,
        background_type,
        {
            "white": white_ratio,
            "blue": blue_ratio,
        },
    )


def validate_applicant_photo_bytes(image_bytes: bytes, include_embedding: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    metrics: dict[str, Any] = {
        "opencv_version": cv2.__version__,
        "face_detector": FACE_DETECTOR_NAME,
    }

    if ENGINE.detector is None:
        return {
            "valid": False,
            "errors": [ENGINE.detector_error or "Face detector is unavailable."],
            "metrics": metrics,
        }

    image = decode_image(image_bytes)

    if image is None:
        return {
            "valid": False,
            "errors": ["Unable to read uploaded photo."],
            "metrics": metrics,
        }

    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height
    aspect_ratio = image_width / image_height if image_height > 0 else 0.0

    metrics["image_width"] = image_width
    metrics["image_height"] = image_height
    metrics["aspect_ratio"] = round(aspect_ratio, 4)

    if not MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
        add_error(
            errors,
            "Photo must be portrait size with width / height between "
            f"{MIN_ASPECT_RATIO:.2f} and {MAX_ASPECT_RATIO:.2f}. "
            f"Current ratio: {aspect_ratio:.2f} ({image_width}x{image_height}).",
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = ENGINE.detect(image, MIN_FACE_SCORE)
    face_count = len(faces)
    metrics["face_count"] = face_count

    face_box: tuple[int, int, int, int] | None = None
    embedding: str | None = None

    if face_count == 0:
        add_error(errors, "No face detected in applicant photo.")
    elif face_count > 1:
        add_error(errors, "Only one face is allowed in applicant photo.")
    else:
        face = faces[0].row

        face_x = int(round(float(face[0])))
        face_y = int(round(float(face[1])))
        face_width = int(round(float(face[2])))
        face_height = int(round(float(face[3])))
        face_score = float(face[14])

        right_eye = (float(face[4]), float(face[5]))
        left_eye = (float(face[6]), float(face[7]))
        nose = (float(face[8]), float(face[9]))
        right_mouth = (float(face[10]), float(face[11]))
        left_mouth = (float(face[12]), float(face[13]))

        # Clamp the box for safe image indexing/background exclusion.
        face_x = max(0, min(face_x, image_width - 1))
        face_y = max(0, min(face_y, image_height - 1))
        face_width = max(1, min(face_width, image_width - face_x))
        face_height = max(1, min(face_height, image_height - face_y))
        face_box = (face_x, face_y, face_width, face_height)

        face_area = face_width * face_height
        face_area_ratio = face_area / image_area if image_area > 0 else 0.0
        face_height_ratio = face_height / image_height if image_height > 0 else 0.0

        face_center_x = face_x + face_width / 2.0
        face_center_y = face_y + face_height / 2.0
        x_offset = abs(face_center_x - image_width / 2.0) / image_width
        y_offset = abs(face_center_y - image_height / 2.0) / image_height

        head_roll = calculate_head_roll(left_eye, right_eye)
        eye_distance = math.dist(left_eye, right_eye)
        eye_mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        mouth_mid = ((left_mouth[0] + right_mouth[0]) / 2.0, (left_mouth[1] + right_mouth[1]) / 2.0)

        # Offsets are measured along the eye line, so a slight head tilt
        # (already checked above) is not mistaken for a turned head.
        nose_eye_offset = sideways_offset(nose, eye_mid, left_eye, right_eye, eye_distance)
        mouth_eye_offset = sideways_offset(mouth_mid, eye_mid, left_eye, right_eye, eye_distance)

        metrics["face"] = {
            "x": face_x,
            "y": face_y,
            "width": face_width,
            "height": face_height,
            "score": round(face_score, 4),
            "area_ratio": round(face_area_ratio, 4),
            "height_ratio": round(face_height_ratio, 4),
            "center_offset_x": round(x_offset, 4),
            "center_offset_y": round(y_offset, 4),
        }
        metrics["landmarks"] = {
            "right_eye": [round(right_eye[0], 2), round(right_eye[1], 2)],
            "left_eye": [round(left_eye[0], 2), round(left_eye[1], 2)],
            "nose": [round(nose[0], 2), round(nose[1], 2)],
            "right_mouth": [round(right_mouth[0], 2), round(right_mouth[1], 2)],
            "left_mouth": [round(left_mouth[0], 2), round(left_mouth[1], 2)],
        }
        metrics["pose"] = {
            "head_roll_degrees": round(head_roll, 2),
            "nose_eye_midpoint_offset_ratio": round(nose_eye_offset, 4),
            "mouth_eye_midpoint_offset_ratio": round(mouth_eye_offset, 4),
        }

        if face_area_ratio < MIN_FACE_AREA_RATIO:
            add_error(errors, "Face is too small in the photo.")

        if face_height_ratio < MIN_FACE_HEIGHT_RATIO:
            add_error(errors, "Face is too far from the camera.")
        elif face_height_ratio > MAX_FACE_HEIGHT_RATIO:
            add_error(errors, "Face is too close to the camera.")

        if (
            x_offset > MAX_FACE_CENTER_OFFSET_X
            or y_offset > MAX_FACE_CENTER_OFFSET_Y
        ):
            add_error(errors, "Face must be centered in the photo.")

        if abs(head_roll) > MAX_HEAD_ROLL_DEGREES:
            add_error(errors, "Applicant face must not be tilted.")

        # This is a lightweight front-facing approximation, not full 3D pose.
        # Nose and mouth should remain close to the midpoint between the eyes.
        if (
            nose_eye_offset > MAX_NOSE_EYE_OFFSET_RATIO
            or mouth_eye_offset > MAX_MOUTH_EYE_OFFSET_RATIO
        ):
            add_error(errors, "Applicant must face the front of the camera.")

        if include_embedding and ENGINE.recognizer is not None:
            embedding = encode_embedding(ENGINE.embed(image, faces[0]))

    # Measure sharpness on the face itself: a plain background has almost no
    # edges and would make a sharp photo look blurry.
    if face_box is not None:
        x, y, w, h = face_box
        sharpness = blur_score(gray[y:y + h, x:x + w])
    else:
        sharpness = blur_score(gray)
    metrics["blur_score"] = round(sharpness, 2)

    if sharpness < MIN_BLUR_SCORE:
        add_error(errors, "Applicant photo is too blurry.")

    background_valid, background_ratio, background_type, background_ratios = (
        check_background_white_or_blue(image, face_box)
    )

    metrics["background"] = {
        "type": background_type,
        "dominant_ratio": round(background_ratio, 4),
        "white_ratio": round(background_ratios["white"], 4),
        "blue_ratio": round(background_ratios["blue"], 4),
        "required_ratio": MIN_BACKGROUND_RATIO,
    }

    if not background_valid:
        add_error(errors, "Photo background must be plain white or blue.")

    result: dict[str, Any] = {
        "valid": len(errors) == 0,
        "errors": errors,
        "metrics": metrics,
    }
    if include_embedding:
        result["embedding"] = embedding
        result["embedding_model"] = RECOGNITION_MODEL_ID
    return result


def face_summary(face: Face) -> dict[str, Any]:
    x, y, w, h = face.box
    return {"x": round(x), "y": round(y), "width": round(w), "height": round(h), "score": round(face.score, 4)}


def embed_main_face(image_bytes: bytes, label: str, min_score: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Embedding of the main face in a photo. Raises ValueError with a message for the user."""
    image = decode_image(image_bytes)
    if image is None:
        raise ValueError(f"Unable to read {label}.")

    faces = ENGINE.detect(image, min_score)
    metrics: dict[str, Any] = {"face_count": len(faces)}
    if not faces:
        raise ValueError(f"No face detected in {label}.")
    if len(faces) > 1 and faces[1].area >= faces[0].area * SECOND_FACE_AREA_RATIO:
        raise ValueError(f"More than one face found in {label}.")

    metrics["face"] = face_summary(faces[0])
    return ENGINE.embed(image, faces[0]), metrics


def recognition_unavailable() -> JSONResponse | None:
    if ENGINE.detector is None or ENGINE.recognizer is None:
        return error_response(
            ENGINE.detector_error or ENGINE.recognizer_error or "Face models are unavailable.", 503
        )
    return None


@app.exception_handler(BadRequest)
def handle_bad_request(_request: Any, exc: BadRequest) -> JSONResponse:
    return error_response(exc.message, exc.status_code)


@app.get("/health")
def health() -> JSONResponse:
    payload, status_code = build_health_payload()

    return JSONResponse(payload, status_code=status_code)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Interactive demo page: validate a photo or compare two faces in the browser."""
    return HTMLResponse((STATIC_DIR / "demo.html").read_text(encoding="utf-8"))


# Handlers are plain `def` so FastAPI runs the CPU-heavy OpenCV work in its
# thread pool instead of blocking the event loop for every other request.
@app.post("/validate-applicant-photo")
def validate_applicant_photo(
    applicant_photo: UploadFile = File(...),
    include_embedding: bool = False,
) -> JSONResponse:
    image_bytes = read_upload(applicant_photo, "applicant photo")
    return JSONResponse(validate_applicant_photo_bytes(image_bytes, include_embedding))


@app.post("/face-embedding")
def face_embedding(photo: UploadFile = File(...)) -> JSONResponse:
    """Enrolment: returns a face template the caller stores and later passes to /verify-face."""
    if (unavailable := recognition_unavailable()) is not None:
        return unavailable

    image_bytes = read_upload(photo, "photo")
    try:
        feature, metrics = embed_main_face(image_bytes, "photo", MIN_REFERENCE_FACE_SCORE)
    except ValueError as exc:
        return JSONResponse({"valid": False, "errors": [str(exc)], "metrics": {}})

    return JSONResponse({
        "valid": True,
        "errors": [],
        "embedding": encode_embedding(feature),
        "embedding_model": RECOGNITION_MODEL_ID,
        "metrics": metrics,
    })


@app.post("/verify-face")
def verify_face(
    applicant_photo: UploadFile = File(...),
    reference_photo: UploadFile | None = File(None),
    reference_embedding: str | None = Form(None),
) -> JSONResponse:
    """1:1 check that the applicant is the same person as a reference photo or stored embedding."""
    if (unavailable := recognition_unavailable()) is not None:
        return unavailable

    if (reference_photo is None) == (not reference_embedding):
        raise BadRequest("Send exactly one of reference_photo or reference_embedding.")

    applicant_bytes = read_upload(applicant_photo, "applicant photo")
    metrics: dict[str, Any] = {"face_recognizer": FACE_RECOGNIZER_NAME, "embedding_model": RECOGNITION_MODEL_ID}

    if reference_embedding:
        reference = decode_embedding(reference_embedding)
        if reference is None:
            raise BadRequest(f"reference_embedding must be a base64 {RECOGNITION_MODEL_ID} embedding from /face-embedding.")
    else:
        reference_bytes = read_upload(reference_photo, "reference photo")  # type: ignore[arg-type]
        try:
            reference, metrics["reference"] = embed_main_face(reference_bytes, "reference photo", MIN_REFERENCE_FACE_SCORE)
        except ValueError as exc:
            return JSONResponse({"match": False, "similarity": None, "threshold": MATCH_THRESHOLD, "errors": [str(exc)], "metrics": metrics})

    try:
        applicant, metrics["applicant"] = embed_main_face(applicant_bytes, "applicant photo", MIN_REFERENCE_FACE_SCORE)
    except ValueError as exc:
        return JSONResponse({"match": False, "similarity": None, "threshold": MATCH_THRESHOLD, "errors": [str(exc)], "metrics": metrics})

    similarity = cosine_similarity(applicant, reference)
    return JSONResponse({
        "match": similarity >= MATCH_THRESHOLD,
        "similarity": round(similarity, 4),
        "threshold": MATCH_THRESHOLD,
        "errors": [],
        "metrics": metrics,
    })
