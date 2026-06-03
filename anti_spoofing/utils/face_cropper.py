# utils/face_cropper.py
"""Face detection and scaled crop — supports multiple faces per frame."""

import numpy as np
from uniface import RetinaFace

from .image_preprocess import crop_face, xyxy2xywh

detector = RetinaFace()

_SCALE = 2.7
_OUT_W = 80
_OUT_H = 80


def get_all_face_crops(frame: np.ndarray, confidence: float = 0.5) -> list[tuple]:
    """Detect all faces and return scaled crops + xyxy bboxes for each.

    Args:
        frame:      BGR frame from OpenCV.
        confidence: Minimum detection confidence to accept a face.

    Returns:
        List of (face_patch, (x1, y1, x2, y2)) tuples.
        Each face_patch is (80, 80, 3) BGR, ready for the model.
        Returns an empty list if no faces are detected.
    """
    faces = detector.detect(frame)
    faces = [f for f in faces if f.confidence >= confidence]

    if not faces:
        return []

    results = []
    for face in faces:
        bbox_xyxy = face.bbox
        bbox_xywh = xyxy2xywh(bbox_xyxy).astype(int).tolist()
        face_patch = crop_face(frame, bbox_xywh, scale=_SCALE, out_w=_OUT_W, out_h=_OUT_H)
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        results.append((face_patch, (x1, y1, x2, y2)))

    return results


def get_face_crop(frame: np.ndarray, confidence: float = 0.5):
    """Single-face convenience wrapper. Returns first face only.

    Returns:
        (face_patch, (x1, y1, x2, y2)) or (None, None).
    """
    crops = get_all_face_crops(frame, confidence)
    if not crops:
        return None, None
    return crops[0]