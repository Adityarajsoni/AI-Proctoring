# engine/proctoring_engine.py
"""
ProctoringEngine
────────────────
Top-level orchestrator.  Owns all four AI services, a face detector
for cropping, a Scheduler, a ViolationEngine, SessionManager, and
EventLogger.

Changes from the previous version
───────────────────────────────────
1. ViolationEngine is now stateful — engine.process_frame no longer needs
   to track "last results" for risk calculation purposes; ViolationEngine
   owns that state.

2. process_frame passes the full EvaluationResult to both SessionManager
   and EventLogger using their new typed APIs.  The old per-violation loop
   in process_frame is gone — EventLogger.log_frame_result() handles it.

3. SessionManager.update() signature has changed:
     OLD: violations, frame_risk_score, anti_spoof, verification, gaze, object_detection
     NEW: violations, instant_risk, session_risk, warning_level,
          new_incidents, violation_details
   The raw service dicts are no longer stored in SessionManager — they are
   purely an EventLogger concern when fine-grained evidence is needed.

4. result["cumulative_risk"] is now a bounded 0–100 session_risk value, not
   an unbounded sum.  The key is kept for backward compatibility.

5. ViolationEngine.reset() is called inside reset_session() to clear its
   internal decay/frequency state alongside the session state.

6. process_frame result dict includes new keys:
     "instant_risk"   — current frame's decayed risk (0–100)
     "session_risk"   — EMA session risk (0–100)
     "warning_level"  — int 0–3
     "incidents"      — list of new incident dicts raised this frame
   The old "frame_risk_score" key maps to instant_risk for compat.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from uniface import RetinaFace

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.scheduler import Scheduler
from violation.violation_engine import ViolationEngine, EvaluationResult
from storage.session_manager import SessionManager
from storage.event_logger import EventLogger

from anti_spoofing.utils.image_preprocess import crop_face as _scaled_crop_face, xyxy2xywh
from anti_spoofing.services.anti_spoof_service import AntiSpoofService
from face_verification.services.face_verification_service import FaceVerificationService
from gaze_tracking.services.gaze_tracking_service import GazeTrackingService
from object_detection.object_detection.services.object_detection_service import ObjectDetectionService

logger = logging.getLogger(__name__)

_FACE_CONF = 0.5


class ProctoringEngine:
    """
    Orchestrates all proctoring services for one student session.

    Usage
    ─────
        engine = ProctoringEngine(student_id="STUDENT001")
        engine.enroll(reference_image)

        while exam_running:
            result = engine.process_frame(frame)
            # result["violations"]    → list of active violation codes
            # result["instant_risk"]  → 0–100, current frame risk
            # result["session_risk"]  → 0–100, EMA session risk
            # result["warning_level"] → int 0–3
            # result["incidents"]     → new incidents this frame

        summary = engine.end_session()
    """

    def __init__(
        self,
        student_id: str = "unknown",
        anti_spoof_model_path: str = "anti_spoofing/models/MiniFASNetV2.pth",
    ) -> None:
        self.student_id = student_id

        logger.info("ProctoringEngine initialising  student=%s", student_id)

        # ── AI Services ───────────────────────────────────────────────────
        self._anti_spoof      = AntiSpoofService(anti_spoof_model_path)
        self._face_verifier   = FaceVerificationService()
        self._gaze_tracker    = GazeTrackingService(student_id=student_id)
        self._object_detector = ObjectDetectionService()

        # ── Face detector for crop extraction ────────────────────────────
        self._face_detector = RetinaFace()

        # ── Orchestration layer ───────────────────────────────────────────
        self._scheduler  = Scheduler()
        self._violations = ViolationEngine()
        self._session    = SessionManager(student_id=student_id)
        self._logger     = EventLogger(student_id=student_id)

        # ── Cached last service results ───────────────────────────────────
        # ViolationEngine owns risk state; these are only cached so that
        # frames where a service doesn't run still pass a valid dict.
        self._last_anti_spoof:   dict = _empty_anti_spoof()
        self._last_verification: dict = _empty_verification()
        self._last_gaze:         dict = _empty_gaze()
        self._last_object:       dict = _empty_object()

        logger.info("ProctoringEngine ready  student=%s", student_id)

    # ------------------------------------------------------------------ #
    # Enrollment                                                           #
    # ------------------------------------------------------------------ #

    def enroll(self, reference_image: np.ndarray) -> dict:
        """
        Register the student's face for verification.
        Must be called once before process_frame().
        """
        result = self._face_verifier.enroll(reference_image)
        logger.info(
            "Enrollment  student=%s  success=%s  msg=%s",
            self.student_id, result["success"], result["message"],
        )
        return result

    # ------------------------------------------------------------------ #
    # Main frame processing                                                #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one camera frame through all scheduled services.

        Returns
        -------
        dict with keys:
            frame_index, anti_spoof, verification, gaze, object_detection,
            violations, instant_risk, session_risk, warning_level,
            frame_risk_score (=instant_risk, legacy alias),
            cumulative_risk  (=session_risk, legacy alias),
            incidents, session
        """
        if frame is None or frame.size == 0:
            logger.warning("process_frame: empty frame received")
            return self._empty_engine_result()

        self._scheduler.tick()
        frame_idx = self._scheduler.frame_index

        # ── Face crop ─────────────────────────────────────────────────────
        face_crop = self._extract_face_crop(frame)

        # ── Gaze (every frame) ────────────────────────────────────────────
        if self._scheduler.should_run("gaze"):
            self._last_gaze = self._gaze_tracker.track(frame)

        # ── Anti-spoofing (every 5 frames) ────────────────────────────────
        if self._scheduler.should_run("anti_spoof"):
            self._last_anti_spoof = (
                self._anti_spoof.predict(face_crop)
                if face_crop is not None
                else _empty_anti_spoof()
            )

        # ── Object detection (every 5 frames) ─────────────────────────────
        if self._scheduler.should_run("object"):
            self._last_object = self._object_detector.detect(frame)

        # ── Face verification (every N frames per scheduler) ──────────────
        if self._scheduler.should_run("verification"):
            if self._face_verifier.is_enrolled():
                self._last_verification = self._face_verifier.verify(frame)

        # ── Violation evaluation ──────────────────────────────────────────
        eval_result: EvaluationResult = self._violations.evaluate(
            anti_spoof_result   = self._last_anti_spoof,
            verification_result = self._last_verification,
            gaze_result         = self._last_gaze,
            object_result       = self._last_object,
            # Future monitors: pass None here until services are available.
            # audio_result         = self._last_audio,
            # secondary_cam_result = self._last_secondary_cam,
            # screen_result        = self._last_screen,
            # browser_result       = self._last_browser,
        )

        # ── Session update ────────────────────────────────────────────────
        self._session.update(
            frame_index       = frame_idx,
            violations        = eval_result.violations,
            instant_risk      = eval_result.instant_risk,
            session_risk      = eval_result.session_risk,
            warning_level     = eval_result.warning_level,
            new_incidents     = eval_result.new_incidents,
            violation_details = eval_result.details,
        )

        # ── Evidence logging ──────────────────────────────────────────────
        # Log warning changes if the warning level advanced this frame.
        prev_warning = getattr(self, "_last_warning_level", eval_result.warning_level)
        if eval_result.warning_level.value > prev_warning.value:
            self._logger.log_warning(
                frame_index    = frame_idx,
                warning_level  = eval_result.warning_level,
                previous_level = prev_warning,
                session_risk   = eval_result.session_risk,
            )
        self._last_warning_level = eval_result.warning_level

        # Log violations and incidents.
        self._logger.log_frame_result(frame_index=frame_idx, result=eval_result)

        return {
            # ── Service outputs ───────────────────────────────────────────
            "frame_index":       frame_idx,
            "anti_spoof":        self._last_anti_spoof,
            "verification":      self._last_verification,
            "gaze":              self._last_gaze,
            "object_detection":  self._last_object,
            # ── Violation / risk ──────────────────────────────────────────
            "violations":        eval_result.violations,
            "instant_risk":      eval_result.instant_risk,
            "session_risk":      eval_result.session_risk,
            "warning_level":     eval_result.warning_level.value,
            "incidents":         [
                {
                    "violation_code": inc.violation_code,
                    "severity":       inc.severity.value,
                    "occurrences":    inc.occurrences,
                    "session_risk":   inc.session_risk,
                }
                for inc in eval_result.new_incidents
            ],
            # ── Legacy aliases (keep downstream callers working) ──────────
            "frame_risk_score":  eval_result.instant_risk,
            "cumulative_risk":   eval_result.session_risk,
            # ── Session snapshot ──────────────────────────────────────────
            "session":           self._session.snapshot(),
        }

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def end_session(self) -> dict:
        """Finalise the session and return the complete summary."""
        gaze_summary    = self._gaze_tracker.end_session()
        session_summary = self._session.finalise()

        self._logger.log_session_end(session_summary)
        self._gaze_tracker.shutdown()
        self._logger.close()

        logger.info(
            "Session ended  student=%s  peak_session_risk=%.1f  incidents=%d",
            self.student_id,
            session_summary["peak_session_risk"],
            session_summary["incident_count"],
        )

        return {
            "session":      session_summary,
            "gaze_summary": gaze_summary,
        }

    def reset_session(self) -> None:
        """Reset all session state without reloading models."""
        self._scheduler.reset()
        self._violations.reset()          # clears decay / frequency state
        self._gaze_tracker.reset_session()
        self._session.reset()
        self._logger.reset()
        self._last_anti_spoof    = _empty_anti_spoof()
        self._last_verification  = _empty_verification()
        self._last_gaze          = _empty_gaze()
        self._last_object        = _empty_object()
        self._last_warning_level = None
        logger.info("ProctoringEngine session reset  student=%s", self.student_id)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_face_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face and return the 80×80 scaled crop that
        AntiSpoofService expects.
        """
        try:
            faces = self._face_detector.detect(frame)
            faces = [f for f in faces if f.confidence >= _FACE_CONF]
            if not faces:
                return None

            face = max(faces, key=lambda f: (
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ))

            bbox_xywh = xyxy2xywh(face.bbox).astype(int).tolist()
            return _scaled_crop_face(
                image=frame, bbox=bbox_xywh, scale=2.7, out_w=80, out_h=80,
            )
        except Exception as exc:
            logger.warning("Face crop failed: %s", exc)
            return None

    def _empty_engine_result(self) -> dict:
        snap = self._session.snapshot()
        return {
            "frame_index":       self._scheduler.frame_index,
            "anti_spoof":        _empty_anti_spoof(),
            "verification":      _empty_verification(),
            "gaze":              _empty_gaze(),
            "object_detection":  _empty_object(),
            "violations":        [],
            "instant_risk":      0.0,
            "session_risk":      snap["session_risk"],
            "warning_level":     snap["warning_level"],
            "incidents":         [],
            "frame_risk_score":  0.0,
            "cumulative_risk":   snap["session_risk"],
            "session":           snap,
        }


# ── Safe empty results ────────────────────────────────────────────────────────

def _empty_anti_spoof() -> dict:
    return {"label": "UNKNOWN", "confidence": 0.0, "is_live": True,
            "raw_label": "Unknown", "raw_score": 0.0, "probabilities": []}

def _empty_verification() -> dict:
    return {"verified": True, "similarity": 0.0,
            "threshold": 0.50, "confidence": 0.0}

def _empty_gaze() -> dict:
    return {"looking_away": False, "gaze_direction": "CENTER",
            "state": "focused", "yaw": 0.0, "pitch": 0.0,
            "risk_score": 0.0, "event": None, "face_detected": False,
            "calibrated": False, "is_attentive": True}

def _empty_object() -> dict:
    return {"phone_detected": False, "multiple_persons": False,
            "no_person": True, "violations": [], "objects": [],
            "violation_detected": False, "laptop_detected": False,
            "person_count": 0, "laptop_count": 0}