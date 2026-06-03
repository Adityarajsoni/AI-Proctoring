

import logging
from typing import Optional

import numpy as np

from .gaze_estimator   import estimate_gaze, GazeResult
from .attention_scorer import AttentionScorer, AttentionScore
from .event_generator  import build_event
from .models.gaze_model import close_face_mesh

logger = logging.getLogger(__name__)


class GazeTracker:
    """
    One instance per student.
    Call process_frame() at 2 FPS throughout the exam.
    """

    # Max consecutive no_face frames before we accept it as real absence
    # At 2 FPS: 1 frame = 0.5 second holdover — brief blip tolerance only
    MAX_HOLDOVER = 1

    def __init__(self, student_id: str):
        self.student_id = student_id
        self._scorer    = AttentionScorer(student_id)
        self._frame_count = 0
        self.last_gaze: Optional[GazeResult] = None
        self._no_face_streak = 0   # consecutive no_face frames
        logger.info("GazeTracker started for student=%s", student_id)

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[dict]:
        """
        Process one frame. Call at 2 FPS (every 500ms).
        Returns Risk Engine event dict if something notable happened, else None.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        self._frame_count += 1
        gaze = estimate_gaze(frame_bgr)

        # Holdover logic 
        # Single no_face frame = MediaPipe blip → skip silently.
        # Multiple consecutive no_face = real absence → update last_gaze
        # so the display also shows no_face (not stale "focused").
        if not gaze.face_detected:
            self._no_face_streak += 1
            if self._no_face_streak <= self.MAX_HOLDOVER and self.last_gaze is not None:
                logger.debug("no_face blip %d/%d — holding last state",
                             self._no_face_streak, self.MAX_HOLDOVER)
                return None
            # Real absence confirmed — update display state too
            self.last_gaze = gaze
        else:
            self._no_face_streak = 0
            self.last_gaze = gaze

        score = self._scorer.update(gaze)
        return build_event(self.student_id, score)

    def end_session(self) -> dict:
        s = self._scorer.summary()
        s["total_frames_processed"] = self._frame_count
        logger.info("GazeTracker ended | student=%s | attention=%.2f",
                    self.student_id, s["attention_rate"])
        return s

    def reset(self):
        self._scorer.reset()
        self._frame_count = 0
        self.last_gaze    = None


def shutdown():
    close_face_mesh()
    logger.info("GazeTracker shutdown.")