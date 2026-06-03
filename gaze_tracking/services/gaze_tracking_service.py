# services/gaze_tracking_service.py
"""
GazeTrackingService
───────────────────
Pure frame-in / result-out service.

Responsibilities:
    - Load MediaPipe FaceMesh once (via the existing singleton) and reuse it.
    - Maintain per-session calibration state (GazeCalibrator).
    - Maintain per-session attention state (AttentionScorer + holdover).
    - track(frame, bbox=None) → structured result dict every call.

Explicitly NOT responsible for:
    - Webcam I/O, cv2.imshow(), drawing
    - Risk Engine routing
    - Session persistence / reports
    - Multi-student orchestration (ProctoringEngine's job)
"""

import logging
from typing import Optional

import numpy as np

from ..gaze_estimator import (
    estimate_gaze,
    GazeCalibrator,
    GazeResult,
    _no_face,
)

from ..attention_scorer import (
    AttentionScorer,
    AttentionScore,
)

from ..event_generator import build_event

from ..models.gaze_model import close_face_mesh

from ..config import (
    STATE_NO_FACE,
    STATE_CALIBRATING,
    STATE_FOCUSED,
)

logger = logging.getLogger(__name__)


class GazeTrackingService:
    """
    One instance per student session.

    Usage
    ─────
        svc = GazeTrackingService(student_id="STUDENT001")

        # Call at 2 FPS throughout the exam
        result = svc.track(frame)
        result = svc.track(frame, bbox=(x1, y1, x2, y2))   # optional crop hint

        # At session end
        summary = svc.end_session()
        svc.shutdown()                # releases MediaPipe resources

    Reset between exam sessions (same student, new exam):
        svc.reset_session()
    """

    # 1-frame holdover: single no_face blip → hold last state silently.
    # Preserved exactly from GazeTracker.MAX_HOLDOVER.
    _MAX_HOLDOVER = 1

    def __init__(self, student_id: str = "unknown") -> None:
        """
        Initialise service for one student session.
        MediaPipe FaceMesh is loaded lazily on the first track() call
        (preserved from the original singleton pattern).
        """
        self.student_id = student_id

        # Session state — all preserved from GazeTracker + GazeEstimator
        self._calibrator      = GazeCalibrator()
        self._scorer          = AttentionScorer(student_id)
        self._frame_count     = 0
        self._no_face_streak  = 0
        self._last_gaze: Optional[GazeResult] = None

        logger.info("GazeTrackingService initialised  student=%s", student_id)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def track(
        self,
        frame: np.ndarray,
        bbox: Optional[tuple] = None,
    ) -> dict:
        """
        Process one frame and return a structured result.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame from the camera.
        bbox : (x1, y1, x2, y2) | None
            Optional face bounding box from another module (e.g. the face
            detector used by AntiSpoofService).  When provided the frame is
            cropped before being passed to MediaPipe so the estimator sees
            only the face region — reduces background interference.
            When None the full frame is used, identical to the original.

        Returns
        -------
        dict with keys:
            looking_away      bool
            gaze_direction    str   – "CENTER" | "LEFT" | "RIGHT" |
                                      "DOWN" | "UP" | "PUPIL_LEFT" |
                                      "PUPIL_RIGHT" | "NO_FACE" | "CALIBRATING"
            state             str   – raw state string from GazeResult
            yaw               float – raw yaw ratio  (-1 to +1)
            pitch             float – raw pitch ratio (-1 to +1)
            yaw_adj           float – calibration-adjusted yaw
            pitch_adj         float – calibration-adjusted pitch
            iris_horizontal   float | None
            iris_vertical     float | None
            face_detected     bool
            calibrated        bool
            calibration_progress str  e.g. "7/10"
            risk_score        float
            risk_reason       str
            consecutive_gaze_away  int
            consecutive_no_face    int
            violations_in_window   int
            is_attentive      bool
            event             dict | None  – Risk Engine event (throttled)
            frame_index       int
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        self._frame_count += 1

        # ── Optional bbox crop ────────────────────────────────────────
        # If a bbox is supplied (from face detector in another module),
        # crop the frame so MediaPipe only sees the face region.
        # Padding of 40 px on each side preserves context landmarks
        # (ears, chin) that the yaw/pitch geometry depends on.
        process_frame = frame
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            h_frame, w_frame = frame.shape[:2]
            pad = 40
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(w_frame, x2 + pad)
            y2c = min(h_frame, y2 + pad)
            process_frame = frame[y1c:y2c, x1c:x2c]

        # ── Gaze estimation (unchanged pipeline) ─────────────────────
        gaze = estimate_gaze(process_frame, self._calibrator)

        # ── Holdover (preserved from GazeTracker.process_frame) ──────
        if not gaze.face_detected:
            self._no_face_streak += 1
            if (self._no_face_streak <= self._MAX_HOLDOVER
                    and self._last_gaze is not None):
                logger.debug(
                    "no_face blip %d/%d — holding last state",
                    self._no_face_streak, self._MAX_HOLDOVER,
                )
                # Return held state without updating scorer
                return self._build_result(
                    gaze=self._last_gaze,
                    score=None,
                    event=None,
                    held=True,
                )
            self._last_gaze = gaze
        else:
            self._no_face_streak = 0
            self._last_gaze = gaze

        # ── Calibration warmup state ──────────────────────────────────
        if not self._calibrator.calibrated:
            # Still collecting neutral samples — don't score yet
            return self._build_result(
                gaze=gaze,
                score=None,
                event=None,
                held=False,
            )

        # ── Attention scoring + event generation ─────────────────────
        score = self._scorer.update(gaze)
        event = build_event(self.student_id, score)

        return self._build_result(gaze=gaze, score=score, event=event, held=False)

    def reset_session(self) -> None:
        """
        Reset all session state between exams.
        MediaPipe model is NOT reloaded — it stays resident.
        """
        self._calibrator     = GazeCalibrator()
        self._scorer.reset()
        self._frame_count    = 0
        self._no_face_streak = 0
        self._last_gaze      = None
        logger.info("GazeTrackingService session reset  student=%s", self.student_id)

    def end_session(self) -> dict:
        """
        Return session summary dict (mirrors GazeTracker.end_session()).
        Call once at exam end before shutdown().
        """
        summary = self._scorer.summary()
        summary["total_frames_processed"] = self._frame_count
        logger.info(
            "GazeTrackingService session ended  student=%s  attention=%.4f",
            self.student_id, summary["attention_rate"],
        )
        return summary

    def shutdown(self) -> None:
        """
        Release MediaPipe FaceMesh resources.
        Call once when the entire proctoring session is over.
        After this, track() must not be called again without
        creating a new GazeTrackingService instance.
        """
        close_face_mesh()
        logger.info("GazeTrackingService shutdown  student=%s", self.student_id)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_result(
        self,
        gaze: GazeResult,
        score: Optional[AttentionScore],
        event: Optional[dict],
        held: bool,
    ) -> dict:
        """Assemble the structured result dict from gaze + score."""
        state = gaze.state

        # Map internal state string → clean gaze_direction for callers
        _dir_map = {
            "focused":       "CENTER",
            "looking_left":  "LEFT",
            "looking_right": "RIGHT",
            "looking_down":  "DOWN",
            "looking_up":    "UP",
            "pupil_left":    "PUPIL_LEFT",
            "pupil_right":   "PUPIL_RIGHT",
            "no_face":       "NO_FACE",
            "calibrating":   "CALIBRATING",
        }

        # During calibration warmup, override displayed state
        if not self._calibrator.calibrated and gaze.face_detected:
            state = STATE_CALIBRATING

        gaze_direction = _dir_map.get(state, state.upper())
        looking_away   = state not in (STATE_FOCUSED, STATE_CALIBRATING, STATE_NO_FACE)

        result = {
            # ── Gaze geometry ─────────────────────────────────────────
            "looking_away":    looking_away,
            "gaze_direction":  gaze_direction,
            "state":           state,
            "yaw":             round(gaze.yaw,       4),
            "pitch":           round(gaze.pitch,     4),
            "roll":            round(gaze.roll,      4),
            "yaw_adj":         round(gaze.yaw_adj,   4),
            "pitch_adj":       round(gaze.pitch_adj, 4),
            "iris_horizontal": round(gaze.iris_horizontal, 4) if gaze.iris_horizontal is not None else None,
            "iris_vertical":   round(gaze.iris_vertical,   4) if gaze.iris_vertical   is not None else None,
            "face_detected":   gaze.face_detected,
            # ── Calibration ───────────────────────────────────────────
            "calibrated":             self._calibrator.calibrated,
            "calibration_progress":   self._calibrator.progress,
            # ── Attention / risk (None when still calibrating) ────────
            "risk_score":             score.risk_score            if score else 0.0,
            "risk_reason":            score.risk_reason           if score else "",
            "consecutive_gaze_away":  score.consecutive_gaze_away if score else 0,
            "consecutive_no_face":    score.consecutive_no_face   if score else 0,
            "violations_in_window":   score.violations_in_window  if score else 0,
            "is_attentive":           score.is_attentive          if score else True,
            # ── Risk Engine event (throttled — None most frames) ──────
            "event":                  event,
            # ── Housekeeping ──────────────────────────────────────────
            "frame_index":            self._frame_count,
            "held_state":             held,
        }

        return result

    def _empty_result(self) -> dict:
        """Returned when frame is None or empty."""
        return {
            "looking_away":    False,
            "gaze_direction":  "NO_FACE",
            "state":           STATE_NO_FACE,
            "yaw":             0.0,
            "pitch":           0.0,
            "roll":            0.0,
            "yaw_adj":         0.0,
            "pitch_adj":       0.0,
            "iris_horizontal": None,
            "iris_vertical":   None,
            "face_detected":   False,
            "calibrated":      self._calibrator.calibrated,
            "calibration_progress": self._calibrator.progress,
            "risk_score":      0.0,
            "risk_reason":     "",
            "consecutive_gaze_away": 0,
            "consecutive_no_face":   0,
            "violations_in_window":  0,
            "is_attentive":    True,
            "event":           None,
            "frame_index":     self._frame_count,
            "held_state":      False,
        }