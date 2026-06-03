# storage/session_manager.py
"""
SessionManager
──────────────
Maintains authoritative session state for one student exam.

Why the old design was broken
──────────────────────────────
1. UNBOUNDED CUMULATIVE RISK
   `_cumulative_risk += frame_risk_score` was called every frame.
   After a 3-hour exam with constant low-level violations the score was
   in the millions — useless as a signal.

2. NO WARNING LIFECYCLE
   There was no concept of warning levels, escalation, or de-escalation.
   Downstream code had no way to ask "has this student been warned twice?"

3. NO INCIDENT RECORD
   Incidents were not stored; they existed only in log files.

4. SILENT HISTORY TRUNCATION
   Once `_HISTORY_MAXLEN` was reached, new frames were silently dropped.
   The `finalise()` summary gave no indication that history was partial.

5. SERVICE CACHE ANTI-PATTERN
   Caching raw service dicts (_latest_anti_spoof etc.) inside the session
   manager mixed storage concerns with service-routing concerns.  The engine
   owns that cache; the session manager should only care about risk signals.

New design
──────────
SESSION RISK            Mirrors ViolationEngine.session_risk (0–100 EMA).
                        Stored here for persistence/finalise; the engine is
                        the source of truth tick-by-tick.

INSTANT RISK            Peak instant risk value seen during the session.
                        Useful for "worst moment" reporting.

WARNING LIFECYCLE       Each warning level (ADVISORY / WARNING / SEVERE) is
                        issued at most once per session unless risk drops by
                        at least `_HYSTERESIS_BAND` below the threshold and
                        re-crosses it.  Prevents warning spam.

INCIDENT STORE          A bounded list of IncidentEvent records.  Passed in
                        by ViolationEngine via update().  Used in finalise()
                        and available to downstream alerting.

VIOLATION FREQUENCY     Per-code running count + first-seen / last-seen times.
                        Used for the finalise() summary and for audit exports.

DECAY-AWARE RISK        session_risk and instant_risk are both 0–100 and
                        bounded.  They are stored as-is (no accumulation).

BOUNDED HISTORY         History is a circular buffer (deque with maxlen).
                        finalise() reports whether it was capped.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from violation.violation_engine import IncidentEvent, WarningLevel

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# Circular frame-history buffer size.  At 30 fps this covers ~5 minutes.
_HISTORY_MAXLEN: int = 9_000

# How many risk points below a threshold the session risk must fall before
# that warning level can be issued again (anti-spam hysteresis).
_HYSTERESIS_BAND: float = 10.0

# Maximum number of incident records kept in memory.
_INCIDENT_MAXLEN: int = 500


# ── Lightweight frame record ───────────────────────────────────────────────────

@dataclass
class FrameRecord:
    """Minimal per-frame snapshot stored in rolling history."""
    frame_index:  int
    violations:   List[str]
    instant_risk: float
    session_risk: float
    warning_level: int         # WarningLevel int value


# ── ViolationStats ─────────────────────────────────────────────────────────────

@dataclass
class ViolationStats:
    """Accumulated statistics for one violation code within a session."""
    code:       str
    count:      int   = 0
    first_seen: Optional[float] = None   # unix timestamp
    last_seen:  Optional[float] = None
    # Total continuous seconds the violation was active (approximate).
    total_duration_s: float = 0.0

    def record(self, now: float, duration_s: float = 0.0) -> None:
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now
        self.count += 1
        self.total_duration_s += duration_s


# ── SessionManager ─────────────────────────────────────────────────────────────

class SessionManager:
    """
    Accumulates session state across all processed frames.

    Usage
    ─────
        sm = SessionManager(student_id="STUDENT001")
        sm.update(
            frame_index=1,
            violations=["PHONE_DETECTED"],
            instant_risk=70.0,
            session_risk=18.5,
            warning_level=WarningLevel.ADVISORY,
            new_incidents=[...],
            violation_details={...},
        )
        snapshot = sm.snapshot()   # lightweight, called every frame
        summary  = sm.finalise()   # called once at session end
    """

    def __init__(self, student_id: str = "unknown") -> None:
        self.student_id  = student_id
        self._start_time = datetime.now(tz=timezone.utc)
        self._reset_state()
        logger.info("SessionManager started  student=%s", student_id)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        frame_index:       int,
        violations:        List[str],
        instant_risk:      float,
        session_risk:      float,
        warning_level:     WarningLevel,
        new_incidents:     List[IncidentEvent],
        violation_details: Dict[str, Any],   # ViolationDetail.to_dict() map
        # Legacy keyword args kept for backward compat with ProctoringEngine
        # that still passes anti_spoof / verification / gaze / object dicts.
        # They are intentionally ignored here — use EventLogger for raw data.
        **_legacy_service_results: Any,
    ) -> None:
        """
        Called once per processed frame by ProctoringEngine.

        Parameters
        ----------
        frame_index       : monotonically increasing frame counter
        violations        : active violation codes this frame
        instant_risk      : bounded 0–100 instant risk from ViolationEngine
        session_risk      : bounded 0–100 EMA session risk
        warning_level     : current WarningLevel enum value
        new_incidents     : IncidentEvent objects raised this frame
        violation_details : dict keyed by violation code with metadata
        """
        import time
        now = time.time()

        self._total_frames  += 1
        self._last_frame_index = frame_index

        # ── Risk tracking ──────────────────────────────────────────────────
        self._latest_instant_risk = instant_risk
        self._latest_session_risk = session_risk
        if instant_risk > self._peak_instant_risk:
            self._peak_instant_risk = instant_risk
        if session_risk > self._peak_session_risk:
            self._peak_session_risk = session_risk

        # ── Violation stats ────────────────────────────────────────────────
        for code in violations:
            detail = violation_details.get(code, {})
            duration_s = (
                detail.duration_s
                if hasattr(detail, "duration_s")
                else float(detail.get("duration_s", 0.0))
            )
            self._violation_stats[code].record(now, duration_s)

        # ── Warning lifecycle ──────────────────────────────────────────────
        new_warning = self._advance_warning(warning_level, session_risk, now)
        if new_warning is not None:
            self._warning_events.append(new_warning)
            logger.warning(
                "Warning issued  student=%s  level=%s  session_risk=%.1f",
                self.student_id, new_warning["level"], session_risk,
            )

        # ── Incident store ─────────────────────────────────────────────────
        for inc in new_incidents:
            if len(self._incidents) < _INCIDENT_MAXLEN:
                self._incidents.append(inc)

        # ── Rolling frame history ──────────────────────────────────────────
        self._history.append(FrameRecord(
            frame_index   = frame_index,
            violations    = list(violations),
            instant_risk  = round(instant_risk, 2),
            session_risk  = round(session_risk, 2),
            warning_level = warning_level.value,
        ))

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a lightweight snapshot of current session state.
        Called every frame by ProctoringEngine for the result dict.
        This must be fast — no heavy computation here.
        """
        return {
            "student_id":          self.student_id,
            "total_frames":        self._total_frames,
            "instant_risk":        round(self._latest_instant_risk, 2),
            "session_risk":        round(self._latest_session_risk, 2),
            "peak_instant_risk":   round(self._peak_instant_risk, 2),
            "peak_session_risk":   round(self._peak_session_risk, 2),
            "warning_level":       self._current_warning_level.value,
            "warning_count":       len(self._warning_events),
            "incident_count":      len(self._incidents),
            "violation_counts":    {k: v.count for k, v in self._violation_stats.items()},
            "last_frame_index":    self._last_frame_index,
        }

    def finalise(self) -> Dict[str, Any]:
        """
        Called once at session end.  Returns the complete session summary.
        """
        import time
        end_time    = datetime.now(tz=timezone.utc)
        duration_s  = round((end_time - self._start_time).total_seconds(), 1)

        violation_summary = {
            code: {
                "count":           stats.count,
                "first_seen":      stats.first_seen,
                "last_seen":       stats.last_seen,
                "total_duration_s": round(stats.total_duration_s, 2),
            }
            for code, stats in self._violation_stats.items()
        }

        incident_summary = [
            {
                "violation_code":  inc.violation_code,
                "severity":        inc.severity.value,
                "occurrences":     inc.occurrences,
                "session_risk":    inc.session_risk,
                "instant_risk":    inc.instant_risk,
                "timestamp":       inc.timestamp,
            }
            for inc in self._incidents
        ]

        history_capped = len(self._history) == _HISTORY_MAXLEN

        summary = {
            "student_id":          self.student_id,
            "session_start":       self._start_time.isoformat(),
            "session_end":         end_time.isoformat(),
            "duration_seconds":    duration_s,
            "total_frames":        self._total_frames,
            # Risk — bounded 0–100
            "peak_instant_risk":   round(self._peak_instant_risk, 2),
            "peak_session_risk":   round(self._peak_session_risk, 2),
            "final_session_risk":  round(self._latest_session_risk, 2),
            # Warnings
            "warning_count":       len(self._warning_events),
            "warning_events":      list(self._warning_events),
            # Incidents
            "incident_count":      len(self._incidents),
            "incidents":           incident_summary,
            # Violations
            "total_unique_violations": len(self._violation_stats),
            "total_violation_events":  sum(
                v.count for v in self._violation_stats.values()
            ),
            "violation_summary":   violation_summary,
            # Metadata
            "history_capped":      history_capped,
        }

        logger.info(
            "Session finalised  student=%s  duration=%.1fs  "
            "peak_session_risk=%.1f  incidents=%d  warnings=%d",
            self.student_id, duration_s,
            self._peak_session_risk,
            len(self._incidents),
            len(self._warning_events),
        )
        return summary

    def reset(self) -> None:
        """Reset between exam sessions without recreating the object."""
        self._start_time = datetime.now(tz=timezone.utc)
        self._reset_state()
        logger.info("SessionManager reset  student=%s", self.student_id)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def session_risk(self) -> float:
        """Current session risk (0–100). Backward-compat alias."""
        return round(self._latest_session_risk, 2)

    @property
    def cumulative_risk(self) -> float:
        """
        Backward-compatible property.
        Returns session_risk (0–100), not an unbounded sum.
        ProctoringEngine uses this in its result dict.
        """
        return self.session_risk

    @property
    def instant_risk(self) -> float:
        return round(self._latest_instant_risk, 2)

    @property
    def warning_level(self) -> WarningLevel:
        return self._current_warning_level

    @property
    def incidents(self) -> List[IncidentEvent]:
        return list(self._incidents)

    # ------------------------------------------------------------------ #
    # Warning lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def _advance_warning(
        self,
        new_level: WarningLevel,
        session_risk: float,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Issue a warning event dict if the warning level has increased.
        Implements hysteresis: once a warning level is issued, session_risk
        must fall below (threshold - HYSTERESIS_BAND) before that level can
        be issued again.

        Returns a warning event dict to append, or None if no new warning.
        """
        from violation.violation_engine import _WARNING_THRESHOLDS

        if new_level.value <= self._current_warning_level.value:
            # Same or lower level — check for de-escalation (hysteresis reset).
            if new_level.value < self._current_warning_level.value:
                # Check whether we've dropped far enough to re-arm the level.
                prev_threshold = _WARNING_THRESHOLDS.get(self._current_warning_level, 0.0)
                if session_risk < prev_threshold - _HYSTERESIS_BAND:
                    # De-escalate: allow re-issuing this level in the future.
                    self._current_warning_level = new_level
            return None

        # New level is higher — issue a warning event.
        self._current_warning_level = new_level
        return {
            "level":        new_level.name,
            "level_int":    new_level.value,
            "session_risk": round(session_risk, 2),
            "timestamp":    now,
            "frame_index":  self._last_frame_index,
        }

    # ------------------------------------------------------------------ #
    # Internal reset                                                       #
    # ------------------------------------------------------------------ #

    def _reset_state(self) -> None:
        self._total_frames:         int   = 0
        self._last_frame_index:     int   = 0
        self._latest_instant_risk:  float = 0.0
        self._latest_session_risk:  float = 0.0
        self._peak_instant_risk:    float = 0.0
        self._peak_session_risk:    float = 0.0
        self._current_warning_level: WarningLevel = WarningLevel.NONE
        self._warning_events:       List[Dict[str, Any]] = []
        self._incidents:            Deque[IncidentEvent] = deque(maxlen=_INCIDENT_MAXLEN)
        self._violation_stats:      Dict[str, ViolationStats] = defaultdict(
            lambda: ViolationStats(code="")  # code filled by record()
        )
        # Override the defaultdict factory to capture code name properly.
        # We use __missing__ via a subclass trick or just set it manually.
        self._violation_stats = _ViolationStatsDict()
        self._history:          Deque[FrameRecord] = deque(maxlen=_HISTORY_MAXLEN)


class _ViolationStatsDict(dict):
    """
    dict subclass that auto-creates ViolationStats with the correct code name
    on first access, avoiding the lambda-capture problem with defaultdict.
    """
    def __missing__(self, key: str) -> ViolationStats:
        stats = ViolationStats(code=key)
        self[key] = stats
        return stats