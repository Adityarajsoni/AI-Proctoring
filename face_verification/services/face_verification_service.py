# services/face_verification_service.py
"""
FaceVerificationService
───────────────────────
Pure embedding-in / result-out service.

Responsibilities:
    - Load InsightFace buffalo_l once and reuse it forever.
    - enroll(image)  → detect face, L2-normalise embedding, store in memory.
    - verify(image)  → detect face, compare against stored embedding, return result.

Explicitly NOT responsible for:
    - Temporal sliding windows  (caller's job — see verify_webcam.py)
    - Session tracking / EMA    (IdentityTracker)
    - SQLite persistence        (EmbeddingStore)
    - Webcam I/O, cv2.imshow    (test_webcam.py)
    - UI / HUD drawing
"""

import logging
import math

import numpy as np
from insightface.app import FaceAnalysis

from ..utils.similarity import cosine_similarity

logger = logging.getLogger(__name__)

# ── Thresholds — preserved exactly from face_verifier.py ────────────────────
VERIFY_THRESHOLD  = 0.50    # cosine similarity cutoff for verified=True
SIGMOID_STEEPNESS = 12.0    # sharpness of the confidence sigmoid
MIN_IMAGE_DIM     = 64      # discard frames smaller than this (px)
# ────────────────────────────────────────────────────────────────────────────


def _sigmoid_confidence(similarity: float) -> float:
    """
    Maps raw cosine similarity → calibrated confidence 0–100.

    Preserved exactly from face_verifier.py::_sigmoid_confidence().

    similarity == VERIFY_THRESHOLD  →  ~50 %
    similarity == 0.70              →  ~97 %
    similarity == 0.35              →  ~ 3 %
    """
    raw = 1.0 / (1.0 + math.exp(-SIGMOID_STEEPNESS * (similarity - VERIFY_THRESHOLD)))
    return round(raw * 100.0, 2)


class FaceVerificationService:
    """
    Stateless verification service with one piece of mutable state:
    the reference embedding set in by enroll().

    Load once, call many times.

    Usage
    ─────
        svc = FaceVerificationService()

        # Enrollment (once per student / session)
        result = svc.enroll(reference_image)
        # {"success": True, "message": "Reference embedding created"}

        # Verification (called on every frame)
        result = svc.verify(live_frame)
        # {
        #     "verified":   True,
        #     "similarity": 0.87,
        #     "threshold":  0.50,
        #     "confidence": 97.14,
        # }
    """

    def __init__(self) -> None:
        """Load buffalo_l once. Reused for all enroll() and verify() calls."""
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))

        # Reference embedding set by enroll(); None until enroll() succeeds.
        self._reference_embedding: np.ndarray | None = None

        logger.info(
            "FaceVerificationService ready  threshold=%.2f", VERIFY_THRESHOLD
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect the largest face in *image* and return its L2-normalised
        512-d embedding, or None if detection/extraction fails.

        Preserved exactly from FaceVerifier.extract_embedding().
        """
        if image is None:
            logger.warning("_extract_embedding: received None image")
            return None

        if image.ndim != 3 or image.shape[2] != 3:
            logger.warning("_extract_embedding: unexpected shape %s", image.shape)
            return None

        h, w = image.shape[:2]
        if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
            logger.warning("_extract_embedding: image too small (%dx%d)", w, h)
            return None

        try:
            faces = self._app.get(image)
        except Exception as exc:
            logger.error("InsightFace inference error: %s", exc)
            return None

        if not faces:
            logger.debug("_extract_embedding: no face detected")
            return None

        # Pick the largest detected face — same as original
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        emb  = largest.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0.0:
            logger.warning("_extract_embedding: zero-norm embedding, discarding")
            return None

        return emb / norm   # unit-sphere → cosine == dot product (fast path)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def enroll(self, image: np.ndarray) -> dict:
        """
        Detect a face in *image*, extract its embedding, and store it as
        the reference for all subsequent verify() calls.

        Parameters
        ----------
        image : np.ndarray
            BGR image containing the student's face (registration photo or
            a clean webcam capture).

        Returns
        -------
        {
            "success": True | False,
            "message": str
        }
        """
        emb = self._extract_embedding(image)

        if emb is None:
            logger.warning("enroll: no face detected — enrollment rejected")
            return {
                "success": False,
                "message": "No face detected in the provided image",
            }

        self._reference_embedding = emb
        logger.info("enroll: reference embedding stored (512-d, L2-normalised)")

        return {
            "success": True,
            "message": "Reference embedding created",
        }

    def verify(self, image: np.ndarray) -> dict:
        """
        Compare the face in *image* against the stored reference embedding.

        Must call enroll() at least once before verify().

        Parameters
        ----------
        image : np.ndarray
            BGR frame or crop containing the student's face.

        Returns
        -------
        {
            "verified":   bool,
            "similarity": float,   # raw cosine similarity 0–1
            "threshold":  float,   # VERIFY_THRESHOLD (0.50)
            "confidence": float,   # sigmoid-calibrated 0–100
        }

        If enroll() has not been called, or face detection fails,
        verified=False and similarity=0.0 are returned — never raises.
        """
        if self._reference_embedding is None:
            logger.error("verify: called before enroll() — no reference embedding")
            return {
                "verified":   False,
                "similarity": 0.0,
                "threshold":  VERIFY_THRESHOLD,
                "confidence": _sigmoid_confidence(0.0),
            }

        live_emb = self._extract_embedding(image)

        if live_emb is None:
            logger.debug("verify: no face detected in live frame")
            return {
                "verified":   False,
                "similarity": 0.0,
                "threshold":  VERIFY_THRESHOLD,
                "confidence": _sigmoid_confidence(0.0),
            }

        similarity = cosine_similarity(self._reference_embedding, live_emb)
        verified   = similarity >= VERIFY_THRESHOLD
        confidence = _sigmoid_confidence(similarity)

        return {
            "verified":   verified,
            "similarity": round(float(similarity), 4),
            "threshold":  VERIFY_THRESHOLD,
            "confidence": confidence,
        }

    def is_enrolled(self) -> bool:
        """Returns True if a reference embedding has been set."""
        return self._reference_embedding is not None

    def clear_enrollment(self) -> None:
        """Remove the stored reference embedding (e.g. between exam sessions)."""
        self._reference_embedding = None
        logger.info("enroll: reference embedding cleared")