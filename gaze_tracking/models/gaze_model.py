# gaze_tracking/models/gaze_model.py
"""
MediaPipe FaceMesh singleton — compatible with mediapipe >= 0.10.

mediapipe 0.10 removed mp.solutions in favour of mp.tasks.
This module supports BOTH APIs so it works on any installed version.
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MAX_NUM_FACES,
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    REFINE_LANDMARKS,
)

logger = logging.getLogger(__name__)

_face_mesh = None


def get_face_mesh():
    """
    Returns a cached FaceMesh instance.
    Compatible with mediapipe < 0.10 (solutions API)
    and mediapipe >= 0.10 (tasks API via legacy shim).
    """
    global _face_mesh
    if _face_mesh is not None:
        return _face_mesh

    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("MediaPipe not installed. Run: pip install mediapipe")

    # ── Try legacy solutions API (mediapipe < 0.10) ───────────────────────
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=REFINE_LANDMARKS,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        logger.info("MediaPipe FaceMesh loaded via solutions API (mediapipe < 0.10)")
        return _face_mesh

    # ── Fallback: mediapipe >= 0.10 tasks API ─────────────────────────────
    # The tasks-based FaceLandmarker is the replacement for FaceMesh.
    # It wraps into a shim so gaze_estimator.py sees the same interface.
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request
        from pathlib import Path

        # Download the face landmarker model if not already present
        model_dir  = Path(__file__).parent / "mediapipe_models"
        model_path = model_dir / "face_landmarker.task"

        if not model_path.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
            logger.info("Downloading face_landmarker.task from %s", url)
            urllib.request.urlretrieve(url, model_path)
            logger.info("Downloaded to %s", model_path)

        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path)
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=MAX_NUM_FACES,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        _face_mesh = _FaceLandmarkerShim(landmarker)
        logger.info("MediaPipe FaceLandmarker loaded via tasks API (mediapipe >= 0.10)")
        return _face_mesh

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MediaPipe FaceMesh/FaceLandmarker: {exc}\n"
            "Try downgrading: pip install mediapipe==0.10.9"
        ) from exc


class _FaceLandmarkerResult:
    """
    Shim that makes FaceLandmarker results look like FaceMesh results.
    gaze_estimator.py accesses: results.multi_face_landmarks[0].landmark
    Each landmark needs .x, .y, .z attributes normalised 0-1.
    FaceLandmarker gives face_landmarks[0] as a list of NormalizedLandmark.
    """

    def __init__(self, task_result, image_width: int, image_height: int):
        self._result = task_result
        self._w = image_width
        self._h = image_height

    @property
    def multi_face_landmarks(self):
        if not self._result.face_landmarks:
            return None
        # Wrap each face's landmark list in a shim
        return [_LandmarkListShim(lms) for lms in self._result.face_landmarks]


class _LandmarkListShim:
    def __init__(self, landmarks):
        self.landmark = landmarks   # list of NormalizedLandmark with .x .y .z


class _FaceLandmarkerShim:
    """
    Wraps mediapipe.tasks FaceLandmarker so it exposes the same
    .process(rgb_image) → result interface as the old FaceMesh.
    """

    def __init__(self, landmarker):
        self._landmarker = landmarker

    def process(self, rgb_image):
        import mediapipe as mp
        h, w = rgb_image.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image,
        )
        task_result = self._landmarker.detect(mp_image)
        return _FaceLandmarkerResult(task_result, w, h)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


def close_face_mesh():
    """Release MediaPipe resources."""
    global _face_mesh
    if _face_mesh is not None:
        try:
            _face_mesh.close()
        except Exception:
            pass
        _face_mesh = None
        logger.info("MediaPipe FaceMesh released.")