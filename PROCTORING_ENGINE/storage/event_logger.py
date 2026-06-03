# storage/event_logger.py
"""
EventLogger
───────────
Append-only, structured JSONL evidence log for one student session.

Why the old design was problematic
────────────────────────────────────
1. FLAT STRUCTURE — every record was a generic "event" with an untyped
   details dict.  Violations, warnings, incidents, and session lifecycle
   events all looked identical, making the log hard to query or audit.

2. NO CONFIDENCE / RISK CONTRIBUTION FIELDS — the record stored a raw
   risk_score number but there was no distinction between "contribution of
   this violation to instant risk" vs "session risk at the time" vs
   "confidence from the underlying model".

3. SAME FILE ACROSS SESSIONS — reset() reopened the same path in append mode,
   so multi-session logs were indistinguishable without external bookkeeping.

4. NO SESSION BOUNDARY MARKERS — there was no START or END record, making
   offline log analysis fragile.

New design
──────────
Four record types, each with a `record_type` discriminator field:

  VIOLATION   — a violation code fired this frame
  WARNING     — a warning level was issued or de-escalated
  INCIDENT    — an incident threshold was crossed
  SESSION     — session START / END lifecycle marker

All records share a common header:
    record_type, timestamp, unix_time, student_id, frame_index

Session-scoped fields added per type:
  VIOLATION:
    violation_code, severity, source,
    instant_risk (contribution, 0–100),
    session_risk (EMA at time of event, 0–100),
    confidence (from underlying service, 0–1),
    duration_s (how long the violation has been active),
    occurrences (rolling-window count),
    details (raw service notes dict)

  WARNING:
    warning_level (int), warning_name (str),
    session_risk, previous_level

  INCIDENT:
    violation_code, severity,
    occurrences (rolling-window count at incident time),
    instant_risk, session_risk

  SESSION:
    event ("START" | "END"),
    session_summary (full summary dict, only on END)

File naming
───────────
  logs/<student_id>_<session_start_iso>.jsonl

  Using a timestamp suffix means each session has its own file — no
  cross-contamination, and reset() always starts a clean file.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from violation.violation_engine import IncidentEvent, WarningLevel, EvaluationResult

logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")

# ── Record type constants ──────────────────────────────────────────────────────
_RT_VIOLATION = "VIOLATION"
_RT_WARNING   = "WARNING"
_RT_INCIDENT  = "INCIDENT"
_RT_SESSION   = "SESSION"


class EventLogger:
    """
    Append-only evidence logger for one student session.

    Usage
    ─────
        el = EventLogger(student_id="STUDENT001")
        # called by ProctoringEngine each frame:
        el.log_frame_result(frame_index=42, result=evaluation_result)
        # called by SessionManager when a warning is issued:
        el.log_warning(frame_index=42, warning_level=WarningLevel.WARNING,
                       previous_level=WarningLevel.ADVISORY, session_risk=52.1)
        # at session end:
        el.log_session_end(session_summary={...})
        el.close()
    """

    def __init__(self, student_id: str = "unknown") -> None:
        self.student_id = student_id
        self._session_start = datetime.now(tz=timezone.utc)
        self._event_count   = 0
        self._path          = self._make_path()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        logger.info("EventLogger opened  path=%s", self._path)
        self._write_session_start()

    # ------------------------------------------------------------------ #
    # Primary frame-level logging                                          #
    # ------------------------------------------------------------------ #

    def log_frame_result(
        self,
        frame_index: int,
        result: EvaluationResult,
    ) -> None:
        """
        Log all violations present in an EvaluationResult in one call.
        Called by ProctoringEngine once per frame (replaces the old loop).

        Parameters
        ----------
        frame_index : camera frame number
        result      : EvaluationResult from ViolationEngine.evaluate()
        """
        for code in result.violations:
            detail = result.details.get(code)
            self._write_violation(
                frame_index   = frame_index,
                violation_code= code,
                severity      = detail.severity.value if detail else "UNKNOWN",
                source        = detail.source         if detail else "webcam",
                instant_risk  = detail.instant_risk   if detail else 0.0,
                session_risk  = result.session_risk,
                duration_s    = detail.duration_s     if detail else 0.0,
                occurrences   = detail.occurrences    if detail else 1,
                confidence    = self._extract_confidence(code, result.source_notes),
                details       = result.source_notes.get(
                    detail.source if detail else "webcam", {}
                ),
            )

        # Log any incidents raised this frame.
        for inc in result.new_incidents:
            self._write_incident(frame_index=frame_index, incident=inc)

    # ------------------------------------------------------------------ #
    # Warning logging                                                      #
    # ------------------------------------------------------------------ #

    def log_warning(
        self,
        frame_index:    int,
        warning_level:  WarningLevel,
        previous_level: WarningLevel,
        session_risk:   float,
    ) -> None:
        """Log a warning-level change event."""
        record = self._base_record(_RT_WARNING, frame_index)
        record.update({
            "warning_level":    warning_level.value,
            "warning_name":     warning_level.name,
            "previous_level":   previous_level.value,
            "previous_name":    previous_level.name,
            "session_risk":     round(session_risk, 2),
        })
        self._write(record)
        logger.info(
            "Warning logged  level=%s  session_risk=%.1f",
            warning_level.name, session_risk,
        )

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def log_session_end(self, session_summary: Dict[str, Any]) -> None:
        """Write a SESSION/END record with the full session summary."""
        record = self._base_record(_RT_SESSION, frame_index=0)
        record.update({
            "event":           "END",
            "session_summary": session_summary,
        })
        self._write(record)

    # ------------------------------------------------------------------ #
    # Backward-compatible single-event API                                 #
    # (kept so existing callers in ProctoringEngine don't break)          #
    # ------------------------------------------------------------------ #

    def log_event(
        self,
        event:       str,
        risk_score:  float,
        frame_index: int = 0,
        details:     Optional[Dict[str, Any]] = None,
        # New optional fields — ignored if not passed
        session_risk:   float = 0.0,
        confidence:     float = 0.0,
        severity:       str   = "UNKNOWN",
        source:         str   = "webcam",
        duration_s:     float = 0.0,
        occurrences:    int   = 1,
    ) -> None:
        """
        Backward-compatible single-violation log call.

        Prefer log_frame_result() for new code — it is richer and avoids
        iterating violations in ProctoringEngine.
        """
        self._write_violation(
            frame_index    = frame_index,
            violation_code = event,
            severity       = severity,
            source         = source,
            instant_risk   = risk_score,
            session_risk   = session_risk,
            duration_s     = duration_s,
            occurrences    = occurrences,
            confidence     = confidence,
            details        = details or {},
        )

    # ------------------------------------------------------------------ #
    # File management                                                      #
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        """Flush and close the log file."""
        try:
            self._fh.flush()
            self._fh.close()
            logger.info(
                "EventLogger closed  path=%s  events=%d",
                self._path, self._event_count,
            )
        except Exception as exc:
            logger.error("EventLogger close failed: %s", exc)

    def reset(self) -> None:
        """
        Close the current log file and open a new one for a fresh session.
        The new file gets its own timestamped name — no cross-contamination.
        """
        self.close()
        self._session_start = datetime.now(tz=timezone.utc)
        self._event_count   = 0
        self._path          = self._make_path()
        self._fh            = self._path.open("a", encoding="utf-8")
        self._write_session_start()
        logger.info("EventLogger reset  new_path=%s", self._path)

    @property
    def log_path(self) -> Path:
        return self._path

    @property
    def event_count(self) -> int:
        return self._event_count

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _make_path(self) -> Path:
        """
        Build a per-session file path.
        Format: logs/<student_id>_<YYYYMMDD_HHMMSS>.jsonl
        """
        ts = self._session_start.strftime("%Y%m%d_%H%M%S")
        return LOGS_DIR / f"{self.student_id}_{ts}.jsonl"

    def _base_record(self, record_type: str, frame_index: int) -> Dict[str, Any]:
        """Return the common header fields shared by all record types."""
        return {
            "record_type": record_type,
            "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
            "unix_time":   time.time(),
            "student_id":  self.student_id,
            "frame_index": frame_index,
        }

    def _write_session_start(self) -> None:
        record = self._base_record(_RT_SESSION, frame_index=0)
        record["event"] = "START"
        self._write(record)

    def _write_violation(
        self,
        frame_index:    int,
        violation_code: str,
        severity:       str,
        source:         str,
        instant_risk:   float,
        session_risk:   float,
        duration_s:     float,
        occurrences:    int,
        confidence:     float,
        details:        Any,
    ) -> None:
        record = self._base_record(_RT_VIOLATION, frame_index)
        record.update({
            "violation_code":  violation_code,
            "severity":        severity,
            "source":          source,
            "instant_risk":    round(instant_risk, 2),
            "session_risk":    round(session_risk, 2),
            "confidence":      round(confidence, 4),
            "duration_s":      round(duration_s, 2),
            "occurrences":     occurrences,
            "details":         details if isinstance(details, dict) else {"raw": str(details)},
        })
        self._write(record)

    def _write_incident(self, frame_index: int, incident: IncidentEvent) -> None:
        record = self._base_record(_RT_INCIDENT, frame_index)
        record.update({
            "violation_code": incident.violation_code,
            "severity":       incident.severity.value,
            "occurrences":    incident.occurrences,
            "instant_risk":   round(incident.instant_risk, 2),
            "session_risk":   round(incident.session_risk, 2),
            "incident_ts":    incident.timestamp,
        })
        self._write(record)
        logger.warning(
            "Incident logged  code=%s  severity=%s  occ=%d",
            incident.violation_code, incident.severity.value, incident.occurrences,
        )

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
            self._event_count += 1
            logger.debug(
                "Record written  type=%s  student=%s",
                record.get("record_type"), self.student_id,
            )
        except Exception as exc:
            logger.error("EventLogger write failed: %s", exc)

    @staticmethod
    def _extract_confidence(
        code: str,
        source_notes: Dict[str, str],
    ) -> float:
        """
        Try to parse a confidence value out of source_notes for the given code.
        Returns 0.0 if not present or not parseable.
        """
        for notes_str in source_notes.values():
            for part in notes_str.split(";"):
                part = part.strip()
                if "confidence=" in part or "conf=" in part:
                    try:
                        return float(part.split("=", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
        return 0.0