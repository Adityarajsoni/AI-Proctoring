
#  Stateful tracker for ONE student session.
#  Converts per-frame GazeResults → AttentionScore with:
#    - Consecutive away tracking   (sustained violations)
#    - Separate no_face tracking   (not mixed with gaze violations)
#    - Rolling window escalation   (repeated pattern)
#    - Pupil-only violation tracking
#
#  gaze violations. Missing face doesn't count toward
#  "looking left" sustained events, and vice versa.


import logging
from collections import deque
from dataclasses import dataclass

from .config import (
    CONSECUTIVE_AWAY_FRAMES,
    NO_FACE_SUSTAINED_FRAMES,
    ESCALATION_COUNT,
    ESCALATION_WINDOW_FRAMES,
    STATE_FOCUSED,
    STATE_NO_FACE,
    RISK_SCORE_FOCUSED,
    RISK_SCORE_SINGLE_AWAY,
    RISK_SCORE_SUSTAINED_AWAY,
    RISK_SCORE_REPEATED_AWAY,
    RISK_SCORE_NO_FACE,
    RISK_SCORE_PUPIL_AWAY,
)

logger = logging.getLogger(__name__)

_GAZE_AWAY_STATES  = {"looking_left", "looking_right",
                      "looking_down",  "looking_up"}
_PUPIL_AWAY_STATES = {"pupil_left", "pupil_right"}


@dataclass
class AttentionScore:
    is_attentive:  bool
    current_state: str

    # Separate counters for each violation type
    consecutive_gaze_away:  int   # head-based look-away frames in a row
    consecutive_no_face:    int   # missing-face frames in a row
    violations_in_window:   int   # sustained events in rolling window

    risk_score:  float
    risk_reason: str

    # Event flags — tell event_generator what to emit this frame
    emit_gaze_sustained:   bool = False
    emit_no_face_sustained: bool = False
    emit_escalation:       bool = False
    emit_pupil_away:       bool = False


class AttentionScorer:
    """One instance per student. Reset between exam sessions."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self._reset()

    def _reset(self):
        self._consec_gaze    = 0    # consecutive head-gaze away frames
        self._consec_noface  = 0    # consecutive no-face frames
        self._total          = 0
        self._away           = 0
        # Rolling window of sustained violation timestamps
        self._sustained_window: deque = deque(maxlen=ESCALATION_WINDOW_FRAMES)

    def update(self, gaze_result) -> AttentionScore:
        self._total += 1
        state = gaze_result.state

        is_gaze_away  = state in _GAZE_AWAY_STATES
        is_pupil_away = state in _PUPIL_AWAY_STATES
        is_no_face    = state == STATE_NO_FACE

        if is_gaze_away or is_pupil_away:
            self._consec_gaze   += 1
            self._consec_noface  = 0   # reset no-face counter
            self._away          += 1
        elif is_no_face:
            self._consec_noface += 1
            self._consec_gaze    = 0   # separate tracking
            self._away          += 1
        else:
            # Focused — reset both counters
            self._consec_gaze   = 0
            self._consec_noface = 0

        # ── Check sustained thresholds 
        emit_gaze_sus   = False
        emit_noface_sus = False
        emit_escalation = False

        # Gaze sustained: fires exactly when threshold crossed
        if self._consec_gaze == CONSECUTIVE_AWAY_FRAMES:
            emit_gaze_sus = True
            self._sustained_window.append(1)
            logger.info("[Attention] student=%s  SUSTAINED gaze away (%d frames) state=%s",
                        self.student_id, self._consec_gaze, state)

        # No-face sustained: separate longer threshold (less aggressive)
        if self._consec_noface == NO_FACE_SUSTAINED_FRAMES:
            emit_noface_sus = True
            logger.info("[Attention] student=%s  SUSTAINED no_face (%d frames)",
                        self.student_id, self._consec_noface)

        # Escalation: many sustained gaze violations in rolling window
        recent = sum(self._sustained_window)
        if recent >= ESCALATION_COUNT and emit_gaze_sus:
            emit_escalation = True
            logger.warning("[Attention] student=%s  ESCALATION — %d violations in window",
                           self.student_id, recent)

        # Risk score 
        if emit_escalation:
            risk, reason = RISK_SCORE_REPEATED_AWAY, f"repeated_look_away_{state}"
        elif emit_gaze_sus:
            risk, reason = RISK_SCORE_SUSTAINED_AWAY, f"sustained_look_away_{state}"
        elif emit_noface_sus:
            risk, reason = RISK_SCORE_NO_FACE, "sustained_face_missing"
        elif is_pupil_away:
            risk, reason = RISK_SCORE_PUPIL_AWAY, f"pupil_deviation_{state}"
        elif is_gaze_away:
            risk, reason = RISK_SCORE_SINGLE_AWAY, f"look_away_{state}"
        else:
            risk, reason = RISK_SCORE_FOCUSED, ""

        return AttentionScore(
            is_attentive         = not (is_gaze_away or is_no_face or is_pupil_away),
            current_state        = state,
            consecutive_gaze_away = self._consec_gaze,
            consecutive_no_face  = self._consec_noface,
            violations_in_window = recent,
            risk_score           = risk,
            risk_reason          = reason,
            emit_gaze_sustained  = emit_gaze_sus,
            emit_no_face_sustained = emit_noface_sus,
            emit_escalation      = emit_escalation,
            emit_pupil_away      = is_pupil_away,
        )

    def reset(self):
        self._reset()
        logger.info("AttentionScorer reset for student=%s", self.student_id)

    @property
    def attention_rate(self) -> float:
        if self._total == 0:
            return 1.0
        return round(1.0 - (self._away / self._total), 4)

    def summary(self) -> dict:
        return {
            "student_id":     self.student_id,
            "total_frames":   self._total,
            "away_frames":    self._away,
            "attention_rate": self.attention_rate,
        }