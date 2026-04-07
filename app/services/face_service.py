"""
Face recognition service using OpenCV's built-in DNN face stack.

Components
----------
YuNet (face_detection_yunet_2023mar.onnx)
    Face detector — locates and aligns face bounding boxes.
SFace (face_recognition_sface_2021dec.onnx)
    Face recogniser — extracts 128-d embeddings; ~99.3 % accuracy on LFW.

Both ONNX models are downloaded automatically to MODELS_DIR on first use.
No TensorFlow or PyTorch installation required.

Similarity threshold: cosine similarity >= FACE_COSINE_THRESHOLD (default 0.363,
matching OpenCV's own recommended value for SFace).
"""

import os
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from .cloudinary_service import resolve_photo_path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
COSINE_THRESHOLD = float(os.getenv("FACE_COSINE_THRESHOLD", "0.363"))
MODELS_DIR = Path(os.getenv("FACE_MODELS_DIR", Path(__file__).parent.parent.parent / "models"))

_DETECTOR_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_RECOGNIZER_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
_DETECTOR_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_RECOGNIZER_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Module-level singletons
_detector: Optional[cv2.FaceDetectorYN] = None
_recognizer: Optional[cv2.FaceRecognizerSF] = None


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _download_model(url: str, dest: Path) -> None:
    """Download a model file if it does not exist yet."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face model to %s …", dest)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    with open(dest, "wb") as fh:
        for chunk in response.iter_content(chunk_size=65536):
            fh.write(chunk)
    logger.info("Model downloaded: %s", dest)


def _load_models() -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    """Lazy-initialise both models on first call."""
    global _detector, _recognizer
    if _detector is not None:
        return _detector, _recognizer  # type: ignore[return-value]

    _download_model(_DETECTOR_MODEL_URL, _DETECTOR_FILE)
    _download_model(_RECOGNIZER_MODEL_URL, _RECOGNIZER_FILE)

    _detector = cv2.FaceDetectorYN.create(
        str(_DETECTOR_FILE),
        config="",
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000,
    )
    _recognizer = cv2.FaceRecognizerSF.create(str(_RECOGNIZER_FILE), config="")
    return _detector, _recognizer


def _read_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image with OpenCV; return None on failure."""
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
    return img


def _detect_faces(img: np.ndarray, detector: cv2.FaceDetectorYN) -> Optional[np.ndarray]:
    """Return the face detections matrix or None when no face is found."""
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None
    return faces


def _get_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Detect the most-confident face in an image and return its SFace embedding.
    Returns None when no face is detected.
    """
    detector, recognizer = _load_models()
    img = _read_image(image_path)
    if img is None:
        return None

    faces = _detect_faces(img, detector)
    if faces is None:
        return None

    # Use the highest-confidence detection (last column = score)
    best_face = faces[np.argmax(faces[:, -1])].reshape(1, -1)
    aligned = recognizer.alignCrop(img, best_face)
    embedding = recognizer.feature(aligned)       # shape (1, 128)
    return embedding


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def validate_face_in_image(image_path: str) -> bool:
    """Return True when at least one face is detected in the image."""
    detector, _ = _load_models()
    img = _read_image(image_path)
    if img is None:
        return False
    return _detect_faces(img, detector) is not None


def count_faces_in_image(image_path: str) -> int:
    """Return the number of faces detected in the image."""
    detector, _ = _load_models()
    img = _read_image(image_path)
    if img is None:
        return 0
    faces = _detect_faces(img, detector)
    return 0 if faces is None else len(faces)


def verify_faces(img1_path: str, img2_path: str) -> tuple[bool, float]:
    """
    Compare two face images.

    Returns
    -------
    (is_match, confidence_score)
        confidence_score is cosine similarity in [0, 1] — higher is closer match.
    """
    _, recognizer = _load_models()

    emb1 = _get_embedding(img1_path)
    emb2 = _get_embedding(img2_path)

    if emb1 is None or emb2 is None:
        return False, 0.0

    score: float = float(recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE))
    confidence = max(0.0, min(1.0, score))
    return confidence >= COSINE_THRESHOLD, confidence


def find_best_match(
    capture_path: str,
    employee_passport_pairs: list[tuple[int, str]],
) -> Optional[tuple[int, float]]:
    """
    Search a capture image against all stored employee passport photos.

    Parameters
    ----------
    capture_path:
        Path to the face image captured from the mobile app.
    employee_passport_pairs:
        List of (employee_db_id, passport_photo_path) tuples.

    Returns
    -------
    (employee_db_id, confidence_score) for the best match, or None if no
    verified match is found.
    """
    best_id: Optional[int] = None
    best_confidence: float = 0.0

    for emp_db_id, passport_path in employee_passport_pairs:
        if not passport_path:
            logger.warning("No passport path for employee db_id=%s — skipping.", emp_db_id)
            continue

        try:
            with resolve_photo_path(passport_path) as local_path:
                is_match, confidence = verify_faces(capture_path, local_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not resolve passport for employee db_id=%s: %s", emp_db_id, exc
            )
            continue

        if is_match and confidence > best_confidence:
            best_confidence = confidence
            best_id = emp_db_id

    if best_id is not None:
        return best_id, best_confidence

    return None
