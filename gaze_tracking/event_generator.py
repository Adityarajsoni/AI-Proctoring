
#  Converts AttentionScore → Risk Engine event dict.
#  Emits events only when something notable happens.
#  No_face events are throttled — only on sustained absence.

import time
import logging
from typing import Optional

from .attention_scorer import AttentionScore

logger = logging.getLogger(__name__)


def build_event(student_id: str,
                score: AttentionScore,
                force: bool = False) -> Optional[dict]:
    """
    Returns a Risk Engine event dict, or None if nothing notable.

    Events emitted only for:
      - Sustained gaze look-away    (not every single away frame)
      - Sustained no-face           (not every missing frame)
      - Escalation pattern
      - Pupil-only deviation        (subtle cheating)
      - force=True                  (periodic heartbeat)

    Single away frames or brief no_face → returns None (not spammy).
    """
    should_emit = (
        force
        or score.emit_gaze_sustained
        or score.emit_no_face_sustained
        or score.emit_escalation
        or score.emit_pupil_away
    )

    if not should_emit:
        return None

    event = {
        # Required by Risk Engine
        "event":     _event_name(score),
        "severity":  round(score.risk_score, 4),
        "timestamp": time.time(),

        # Gaze context
        "student_id":           student_id,
        "gaze_state":           score.current_state,
        "consecutive_away":     score.consecutive_gaze_away,
        "violations_in_window": score.violations_in_window,
        "risk_reason":          score.risk_reason,
        "is_attentive":         score.is_attentive,
    }

    logger.info("[Event] student=%s  event=%-28s  severity=%.2f",
                student_id, event["event"], event["severity"])
    return event


def _event_name(score: AttentionScore) -> str:
    if score.emit_escalation:
        return "gaze_repeated_violation"
    if score.emit_gaze_sustained:
        return "gaze_sustained_away"
    if score.emit_no_face_sustained:
        return "gaze_face_missing_sustained"
    if score.emit_pupil_away:
        return "gaze_pupil_deviation"
    return "gaze_away"