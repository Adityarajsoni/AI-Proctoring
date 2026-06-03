"""
ObjectDetectionService
======================
Pure inference service. No webcam. No display. No state.

Responsibilities:
    Frame → YOLO Inference → Detection Filtering → Violation Extraction → Structured Result

Designed to be owned by ProctoringEngine alongside:
    AntiSpoofService, FaceVerificationService, GazeTrackingService
"""

import numpy as np
from typing import Dict, Any

from object_detection.app.detectors.yolo_detector import YOLODetector
from object_detection.app.config.settings import settings
from object_detection.app.utils.logger import app_logger, error_logger


# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VIOLATION_PHONE_DETECTED      = "PHONE_DETECTED"
VIOLATION_MULTIPLE_PERSONS    = "MULTIPLE_PERSONS"
VIOLATION_NO_PERSON           = "NO_PERSON"
VIOLATION_LAPTOP_DETECTED     = "LAPTOP_DETECTED"
VIOLATION_MULTIPLE_LAPTOPS    = "MULTIPLE_LAPTOPS"


# ─────────────────────────────────────────────────────────────────────────────
# CLASS ID ALIASES  (sourced from settings.TARGET_CLASSES — never hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_PERSON = settings.TARGET_CLASSES[0]   # 0
_CLASS_LAPTOP = settings.TARGET_CLASSES[1]   # 63
_CLASS_PHONE  = settings.TARGET_CLASSES[2]   # 67


class ObjectDetectionService:
    """
    Stateless per-frame object detection service.

    Usage
    -----
        service = ObjectDetectionService()
        result  = service.detect(frame)

    Returns
    -------
        {
            "person_count":       int,
            "multiple_persons":   bool,
            "no_person":          bool,
            "phone_detected":     bool,
            "laptop_detected":    bool,
            "multiple_laptops":   bool,
            "laptop_count":       int,
            "violation_detected": bool,
            "violations":         List[str],
            "objects":            List[{class_id, class_name, confidence, bbox}]
        }
    """

    def __init__(self) -> None:
        app_logger.info("Initializing ObjectDetectionService")
        try:
            self._detector = YOLODetector()
            self._detector.warmup()
            app_logger.info("ObjectDetectionService ready")
        except Exception as e:
            error_logger.exception(f"ObjectDetectionService failed to initialize: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run YOLO inference on a single frame and return a structured result.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR webcam frame (H × W × 3).

        Returns
        -------
        dict
            Structured detection + violation result. See class docstring.
        """
        try:
            raw = self._detector.detect(frame)
            detections = raw["detections"]

            parsed  = self._parse(detections)
            violations = self._extract_violations(parsed)

            return {
                # ── person summary ────────────────────────────────────────
                "person_count":       parsed["person_count"],
                "multiple_persons":   parsed["person_count"] > 1,
                "no_person":          parsed["person_count"] == 0,
                # ── object flags ─────────────────────────────────────────
                "phone_detected":     parsed["phone_detected"],
                "laptop_detected":    parsed["laptop_count"] > 0,
                "multiple_laptops":   parsed["laptop_count"] > 1,
                "laptop_count":       parsed["laptop_count"],
                # ── violations ───────────────────────────────────────────
                "violation_detected": len(violations) > 0,
                "violations":         violations,
                # ── raw detections ────────────────────────────────────────
                "objects":            detections,
            }

        except Exception as e:
            error_logger.exception(f"ObjectDetectionService.detect failed: {e}")
            return self._empty_result()

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _parse(self, detections: list) -> Dict[str, Any]:
        """
        Walk the raw detections list and tally per-class counts/flags.
        Preserves the same class_id comparisons used in ObjectDetector
        and SideFaceDetector.
        """
        person_count   = 0
        laptop_count   = 0
        phone_detected = False

        for det in detections:
            cid = det["class_id"]

            if cid == _CLASS_PERSON:
                person_count += 1

            elif cid == _CLASS_LAPTOP:
                laptop_count += 1

            elif cid == _CLASS_PHONE:
                phone_detected = True

        return {
            "person_count":   person_count,
            "laptop_count":   laptop_count,
            "phone_detected": phone_detected,
        }

    def _extract_violations(self, parsed: Dict[str, Any]) -> list:
        """
        Map parsed counts/flags to violation string constants.

        Rules (identical to ObjectDetector / SideFaceDetector logic):
            - person_count == 0  → NO_PERSON
            - person_count  > 1  → MULTIPLE_PERSONS
            - phone_detected     → PHONE_DETECTED
            - laptop_count  > 0  → LAPTOP_DETECTED
            - laptop_count  > 1  → MULTIPLE_LAPTOPS   (replaces LAPTOP_DETECTED)
        """
        violations = []

        if parsed["person_count"] == 0:
            violations.append(VIOLATION_NO_PERSON)

        if parsed["person_count"] > 1:
            violations.append(VIOLATION_MULTIPLE_PERSONS)

        if parsed["phone_detected"]:
            violations.append(VIOLATION_PHONE_DETECTED)

        if parsed["laptop_count"] > 1:
            violations.append(VIOLATION_MULTIPLE_LAPTOPS)
        elif parsed["laptop_count"] == 1:
            violations.append(VIOLATION_LAPTOP_DETECTED)

        return violations

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """Safe fallback returned on inference errors."""
        return {
            "person_count":       0,
            "multiple_persons":   False,
            "no_person":          True,
            "phone_detected":     False,
            "laptop_detected":    False,
            "multiple_laptops":   False,
            "laptop_count":       0,
            "violation_detected": False,
            "violations":         [],
            "objects":            [],
        }