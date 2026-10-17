"""OpenCV face models: YuNet for detection, SFace for recognition (embeddings)."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DETECTION_MODEL = BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
RECOGNITION_MODEL_ID = "sface_2021dec"

# Large phone photos are downscaled for detection only; landmarks are mapped back.
MAX_DETECTION_SIDE = 1280
# SFace cosine similarity at or above this means "same person" (OpenCV Zoo reference value).
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.363"))
EMBEDDING_SIZE = 128


@dataclass
class Face:
    """One YuNet detection in original-image coordinates."""

    row: np.ndarray  # x, y, w, h, 5 landmark pairs, score

    @property
    def box(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.row[:4])  # type: ignore[return-value]

    @property
    def area(self) -> float:
        return float(self.row[2] * self.row[3])

    @property
    def score(self) -> float:
        return float(self.row[14])


class FaceEngine:
    def __init__(self) -> None:
        # YuNet is stateful (setInputSize / setScoreThreshold), so every call holds a lock.
        self._detector_lock = Lock()
        self._recognizer_lock = Lock()
        self.detector: Any | None = None
        self.recognizer: Any | None = None
        self.detector_error: str | None = None
        self.recognizer_error: str | None = None
        self._load_detector()
        self._load_recognizer()

    def _load_detector(self) -> None:
        if not DETECTION_MODEL.exists():
            self.detector_error = f"YuNet face model is missing. Expected file: {DETECTION_MODEL}"
            return
        if not hasattr(cv2, "FaceDetectorYN"):
            self.detector_error = (
                "This OpenCV build does not expose cv2.FaceDetectorYN. "
                f"Detected OpenCV version: {cv2.__version__}."
            )
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(str(DETECTION_MODEL), "", (320, 320), 0.6, 0.3, 5000)
        except cv2.error as exc:
            self.detector_error = f"Unable to initialize YuNet face detector: {exc}"

    def _load_recognizer(self) -> None:
        if not RECOGNITION_MODEL.exists():
            self.recognizer_error = f"SFace recognition model is missing. Expected file: {RECOGNITION_MODEL}"
            return
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNITION_MODEL), "")
        except cv2.error as exc:
            self.recognizer_error = f"Unable to initialize SFace face recognizer: {exc}"

    def detect(self, image: np.ndarray, min_score: float) -> list[Face]:
        """Detect faces, largest first."""
        if self.detector is None:
            return []

        height, width = image.shape[:2]
        scale = min(1.0, MAX_DETECTION_SIDE / max(height, width))
        small = image if scale == 1.0 else cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )

        with self._detector_lock:
            self.detector.setScoreThreshold(min_score)
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self.detector.detect(small)

        if rows is None:
            return []

        rows = rows.copy()
        rows[:, :14] /= scale
        faces = [Face(row) for row in rows]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """128-d L2-normalised SFace feature for one detected face."""
        if self.recognizer is None:
            raise RuntimeError(self.recognizer_error or "Face recognizer is unavailable.")
        with self._recognizer_lock:
            aligned = self.recognizer.alignCrop(image, face.row)
            feature = self.recognizer.feature(aligned)
        feature = feature.flatten().astype(np.float32)
        return feature / max(float(np.linalg.norm(feature)), 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def encode_embedding(feature: np.ndarray) -> str:
    return base64.b64encode(feature.astype("<f4").tobytes()).decode("ascii")


def decode_embedding(value: str) -> np.ndarray | None:
    """Parse an embedding returned earlier by this API; None if it is malformed."""
    try:
        raw = base64.b64decode(value.strip(), validate=True)
    except (ValueError, TypeError):
        return None
    if len(raw) != EMBEDDING_SIZE * 4:
        return None
    feature = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    return feature if np.all(np.isfinite(feature)) and np.linalg.norm(feature) > 0 else None


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
