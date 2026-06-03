# violation/violation_engine.py
"""
ViolationEngine
───────────────
Stateful per-session rule evaluator.

Why the old design was broken
──────────────────────────────
1. UNBOUNDED RISK ACCUMULATION
   The old engine was truly stateless — it returned `sum(weights)` of
   active violations every frame and handed that number to SessionManager,
   which kept adding it to `_cumulative_risk`.  After a 3-hour exam at
   30 fps a single sustained LOOKING_AWAY (weight 10) would contribute
   10 × 30 × 10800 = 3,240,000 to the cumulative score.  The number was
   meaningless as a risk signal.

2. NO DECAY
   Once a violation appeared it stayed at full weight until it disappeared
   entirely.  There was no model of "risk fading as clean frames accumulate".

3. NO DUAL VIOLATION GUARD
   NO_FACE and LOOKING_AWAY could both fire simultaneously, double-counting
   the same event with additive weights.

4. NO WARNING / INCIDENT LIFECYCLE
   Every call was stateless so there was no way to know "this is the 5th
   PHONE_DETECTED in 30 seconds → escalate to incident".

5. NO NORMALIZED OUTPUT
   Consumers had no bounded signal.  A downstream dashboard showing "risk"
   was displaying a number that could be 0 or 3,000,000 depending on exam
   length.

New design
──────────
INSTANT RISK  (0–100, bounded)
  A decaying signal that spikes when a violation fires and decays
  exponentially toward zero while no violation is active.
  Formula per tick:
    instant_risk = max(instant_risk * (1 - decay_rate * dt), 0)
    if violation active: instant_risk = min(instant_risk + base_weight, 100)

SESSION RISK  (0–100, bounded)
  A smoothed, time-averaged view of instant risk.  Uses an exponential
  moving average so a single noisy frame never pins the session to 100.
  α = 0.05 (slow adaptation — represents "overall exam behaviour").

WARNING LEVELS  (0, 1, 2, 3)
  Driven by session_risk thresholds.  Each level is issued at most once
  unless the session risk drops below the prior threshold and rises again
  (hysteresis built into SessionManager).

INCIDENTS
  When a specific violation fires more times than its `incident_threshold`
  within a rolling 60-second window, ViolationEngine emits an incident
  event that SessionManager records.

FREQUENCY TRACKER
  A rolling-window counter (60 s) tracks how many times each violation
  fired in recent history, used for:
    - escalation (LOOKING_AWAY → GAZE_SUSTAINED_AWAY vocabulary)
    - incident threshold checking

DURATION TRACKER
  For each currently-active violation we record when it started.
  If it is still active next tick, duration accumulates.  Used for:
    - GAZE_SUSTAINED_AWAY upgrade
    - evidence timestamps in EventLogger

NO_FACE / LOOKING_AWAY EXCLUSIVITY
  NO_FACE implies LOOKING_AWAY, so only the more severe of the two fires.

Future monitor sources (audio, dual-camera, screen, browser)
  evaluate() accepts **monitor_results as keyword arguments.
  Each monitor plugin registers its violation codes in violation_rules.py
  and passes its result dict through evaluate().
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from violation.violation_rules import (
    RULES,
    Severity,
    get_decay_rate,
    get_rule,
    get_severity,
    get_weight,
)

logger = logging.getLogger(__name__)


# ── Warning levels ─────────────────────────────────────────────────────────────

class WarningLevel(int, Enum):
    NONE     = 0
    ADVISORY = 1   # session_risk ≥ 25
    WARNING  = 2   # session_risk ≥ 50
    SEVERE   = 3   # session_risk ≥ 75


# Session-risk thresholds that trigger each warning level.
_WARNING_THRESHOLDS: Dict[WarningLevel, float] = {
    WarningLevel.ADVISORY: 25.0,
    WarningLevel.WARNING:  50.0,
    WarningLevel.SEVERE:   75.0,
}

# ── Rolling-window configuration ───────────────────────────────────────────────
_FREQUENCY_WINDOW_SECONDS: float = 60.0


# ── Output dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ViolationDetail:
    """Per-violation metadata returned inside EvaluationResult.details."""
    code:          str
    severity:      Severity
    instant_risk:  float          # contribution this tick (0–100 anchor)
    duration_s:    float          # seconds this violation has been continuously active
    occurrences:   int            # times fired in rolling 60-s window
    source:        str = "webcam" # monitor source


@dataclass
class IncidentEvent:
    """
    Raised when a violation crosses its incident_threshold in the rolling window.
    Passed back to the caller (SessionManager / EventLogger) for recording.
    """
    violation_code:  str
    severity:        Severity
    occurrences:     int           # rolling-window count at time of incident
    session_risk:    float
    instant_risk:    float
    timestamp:       float = field(default_factory=time.time)


@dataclass
class EvaluationResult:
    """
    Full output of ViolationEngine.evaluate().

    Replace the old bare dict with a typed structure.
    Callers that still need a plain dict can call .to_dict().
    """
    # Active violation codes (deduplicated, ordered by severity desc)
    violations:     List[str]

    # Risk signals — both bounded 0–100
    instant_risk:   float          # current frame's decayed+spiked risk
    session_risk:   float          # slow EMA of instant_risk over session

    # Warning level derived from session_risk
    warning_level:  WarningLevel

    # Per-violation metadata
    details:        Dict[str, ViolationDetail]

    # Incidents raised this tick (usually empty; list for safety)
    new_incidents:  List[IncidentEvent]

    # Raw per-source detail strings for logging / HUD
    source_notes:   Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict for callers that expect the old structure."""
        return {
            "violations":    self.violations,
            "instant_risk":  round(self.instant_risk, 2),
            "session_risk":  round(self.session_risk, 2),
            "risk_score":    round(self.instant_risk, 2),   # legacy alias
            "warning_level": self.warning_level.value,
            "details":       self.source_notes,             # legacy alias
            "new_incidents": [
                {
                    "violation_code": inc.violation_code,
                    "severity":       inc.severity.value,
                    "occurrences":    inc.occurrences,
                    "session_risk":   inc.session_risk,
                    "instant_risk":   inc.instant_risk,
                    "timestamp":      inc.timestamp,
                }
                for inc in self.new_incidents
            ],
        }


# ── ViolationEngine ────────────────────────────────────────────────────────────

class ViolationEngine:
    """
    Stateful, per-session violation evaluator.

    Lifecycle
    ─────────
        engine = ViolationEngine()
        result = engine.evaluate(
            anti_spoof_result,
            verification_result,
            gaze_result,
            object_result,
            # future:
            # audio_result=...,
            # secondary_cam_result=...,
            # screen_result=...,
        )
        # result is an EvaluationResult
        # result.to_dict() for backward-compat

    Thread safety
    ─────────────
    Not thread-safe.  ProctoringEngine calls evaluate() from one thread only.
    If you introduce background workers, add a lock here.
    """

    def __init__(self) -> None:
        # ── Instant risk state ─────────────────────────────────────────────
        # Maps violation_code → current instant-risk contribution (0–100).
        # Decays toward 0 each tick; spikes when violation fires.
        self._instant_risk_by_violation: Dict[str, float] = defaultdict(float)

        # ── Session risk (EMA) ─────────────────────────────────────────────
        self._session_risk: float = 0.0
        self._ema_alpha: float    = 0.05   # slow adaptation

        # ── Warning level ──────────────────────────────────────────────────
        self._warning_level: WarningLevel = WarningLevel.NONE

        # ── Rolling-frequency window ───────────────────────────────────────
        # Maps violation_code → deque of unix timestamps (recent firings).
        self._frequency: Dict[str, Deque[float]] = defaultdict(deque)

        # ── Duration tracking ──────────────────────────────────────────────
        # Maps violation_code → (start_time, last_seen_time) when active.
        self._active_since: Dict[str, Tuple[float, float]] = {}

        # ── Incident tracking ──────────────────────────────────────────────
        # Maps violation_code → set of rolling-window counts at which we
        # already raised an incident (prevents re-raising on same spike).
        self._incident_raised_at: Dict[str, int] = defaultdict(int)

        # ── Timing ─────────────────────────────────────────────────────────
        self._last_tick: float = time.monotonic()

        logger.debug("ViolationEngine initialised")

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        anti_spoof_result:    Dict[str, Any],
        verification_result:  Dict[str, Any],
        gaze_result:          Dict[str, Any],
        object_result:        Dict[str, Any],
        # Future monitor results — accept as optional kwargs so the
        # method signature stays stable when new monitors are added.
        audio_result:         Optional[Dict[str, Any]] = None,
        secondary_cam_result: Optional[Dict[str, Any]] = None,
        screen_result:        Optional[Dict[str, Any]] = None,
        browser_result:       Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate all monitor inputs and return a rich EvaluationResult.

        Parameters
        ----------
        anti_spoof_result    : dict from AntiSpoofService
        verification_result  : dict from FaceVerificationService
        gaze_result          : dict from GazeTrackingService
        object_result        : dict from ObjectDetectionService
        audio_result         : (future) dict from AudioMonitorService
        secondary_cam_result : (future) dict from SecondaryCameraService
        screen_result        : (future) dict from ScreenMonitorService
        browser_result       : (future) dict from BrowserMonitorService

        Returns
        -------
        EvaluationResult (call .to_dict() for backward-compat dict)
        """
        now       = time.monotonic()
        wall_time = time.time()
        dt        = max(now - self._last_tick, 0.001)  # seconds since last call
        self._last_tick = now

        # ── Step 1: Collect raw violations from each monitor ───────────────
        raw: List[Tuple[str, str, Dict[str, str]]] = []
        # Each entry: (violation_code, source_label, note_dict)

        raw.extend(self._check_anti_spoof(anti_spoof_result))
        raw.extend(self._check_verification(verification_result))
        raw.extend(self._check_gaze(gaze_result))
        raw.extend(self._check_objects(object_result))

        # Future monitors — only wired when result dicts are provided.
        if audio_result is not None:
            raw.extend(self._check_audio(audio_result))
        if secondary_cam_result is not None:
            raw.extend(self._check_secondary_cam(secondary_cam_result))
        if screen_result is not None:
            raw.extend(self._check_screen(screen_result))
        if browser_result is not None:
            raw.extend(self._check_browser(browser_result))

        # ── Step 2: Deduplicate, resolve exclusions ────────────────────────
        active_codes: List[str] = self._resolve_exclusions(
            [code for code, _, _ in raw]
        )
        active_set = set(active_codes)

        # Build source_notes for logging / HUD
        source_notes: Dict[str, str] = {}
        for code, source, notes in raw:
            if code in active_set:
                source_notes[source] = "; ".join(
                    f"{k}={v}" for k, v in notes.items()
                )

        # ── Step 3: Update frequency + duration trackers ───────────────────
        self._update_frequency(active_codes, wall_time)
        self._update_duration(active_codes, wall_time)

        # ── Step 4: Decay ALL instant-risk buckets ─────────────────────────
        for code in list(self._instant_risk_by_violation.keys()):
            decay = get_decay_rate(code)
            self._instant_risk_by_violation[code] = max(
                self._instant_risk_by_violation[code] * (1.0 - decay * dt),
                0.0,
            )

        # ── Step 5: Spike instant risk for active violations ───────────────
        for code in active_codes:
            spike = get_weight(code)
            current = self._instant_risk_by_violation[code]
            # Spike up to but not beyond 100.
            self._instant_risk_by_violation[code] = min(current + spike, 100.0)

        # ── Step 6: Compute aggregate instant_risk ─────────────────────────
        instant_risk = self._aggregate_instant_risk()

        # ── Step 7: Update session EMA ─────────────────────────────────────
        self._session_risk = (
            self._ema_alpha * instant_risk
            + (1.0 - self._ema_alpha) * self._session_risk
        )
        session_risk = round(min(self._session_risk, 100.0), 2)

        # ── Step 8: Derive warning level ──────────────────────────────────
        warning_level = self._compute_warning_level(session_risk)
        self._warning_level = warning_level

        # ── Step 9: Raise incidents ────────────────────────────────────────
        new_incidents = self._check_incidents(
            active_codes, instant_risk, session_risk, wall_time
        )

        # ── Step 10: Build per-violation details ───────────────────────────
        details: Dict[str, ViolationDetail] = {}
        for code in active_codes:
            rule = get_rule(code)
            severity = rule.severity if rule else Severity.LOW
            start, last = self._active_since.get(code, (wall_time, wall_time))
            duration_s = last - start
            occ = self._rolling_count(code, wall_time)
            details[code] = ViolationDetail(
                code         = code,
                severity     = severity,
                instant_risk = round(self._instant_risk_by_violation.get(code, 0.0), 2),
                duration_s   = round(duration_s, 2),
                occurrences  = occ,
                source       = self._source_for(code),
            )

        # ── Step 11: Sort violations by severity descending ────────────────
        active_codes.sort(
            key=lambda c: get_severity(c).warning_level,
            reverse=True,
        )

        if active_codes:
            logger.info(
                "Violations: %s  instant=%.1f  session=%.1f  warning=%s",
                active_codes, instant_risk, session_risk, warning_level.name,
            )

        return EvaluationResult(
            violations    = active_codes,
            instant_risk  = round(instant_risk, 2),
            session_risk  = session_risk,
            warning_level = warning_level,
            details       = details,
            new_incidents = new_incidents,
            source_notes  = source_notes,
        )

    # ------------------------------------------------------------------ #
    # Per-source checks                                                    #
    # ── Each returns List[Tuple[code, source_label, notes_dict]]         #
    # ── An empty list means "no violations from this source"             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_anti_spoof(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        label   = result.get("label", "UNKNOWN")
        is_live = result.get("is_live", True)
        conf    = result.get("confidence", 0.0)
        # Only fire when the service has made a real determination.
        if label == "UNKNOWN" or is_live:
            return []
        return [("SPOOF_DETECTED", "anti_spoof", {
            "label": label,
            "confidence": f"{conf:.3f}",
        })]

    @staticmethod
    def _check_verification(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        similarity = result.get("similarity", 0.0)
        verified   = result.get("verified", True)
        # Skip the empty default (similarity==0, verified==True).
        if similarity == 0.0 and verified:
            return []
        if verified:
            return []
        return [("FACE_MISMATCH", "verification", {
            "similarity": f"{similarity:.4f}",
        })]

    @staticmethod
    def _check_gaze(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        looking_away  = result.get("looking_away", False)
        gaze_dir      = result.get("gaze_direction", "CENTER")
        gaze_event    = result.get("event")
        gaze_risk     = result.get("risk_score", 0.0)

        out: List[Tuple[str, str, Dict[str, str]]] = []

        # NO_FACE is mutually exclusive with LOOKING_AWAY variants.
        # A missing face already implies "not looking at screen" — reporting
        # both would double-count the same root cause.
        if gaze_dir == "NO_FACE":
            out.append(("NO_FACE", "gaze", {
                "direction": gaze_dir,
                "risk": f"{gaze_risk:.2f}",
            }))
            # Do NOT also append LOOKING_AWAY / GAZE_SUSTAINED_AWAY.
            return out

        if not looking_away:
            return out

        # Map gaze service event to the correct violation tier.
        code = "LOOKING_AWAY"
        if gaze_event:
            event_name = gaze_event.get("event", "")
            if "repeated" in event_name:
                code = "GAZE_ESCALATION"
            elif "sustained" in event_name:
                code = "GAZE_SUSTAINED_AWAY"

        out.append((code, "gaze", {
            "direction": gaze_dir,
            "risk": f"{gaze_risk:.2f}",
        }))
        return out

    @staticmethod
    def _check_objects(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        obj_violations: List[str] = result.get("violations", [])
        objects: List[Any]        = result.get("objects", [])
        obj_labels = ", ".join(
            str(o.get("label", o)) if isinstance(o, dict) else str(o)
            for o in objects
        ) or "unknown"
        return [
            (code, "object_detection", {"objects": obj_labels})
            for code in obj_violations
        ]

    # ── Future monitor checks (stubs — return [] until implemented) ────────

    @staticmethod
    def _check_audio(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        """
        Evaluate AudioMonitorService output.

        Expected result keys (when service is live):
            voice_detected    : bool
            voice_count       : int    (number of distinct speakers)
            confidence        : float
            violations        : list[str]  (codes from violation_rules.py)

        Until AudioMonitorService is implemented, this method returns [].
        """
        violations: List[str] = result.get("violations", [])
        confidence: float     = result.get("confidence", 0.0)
        return [
            (code, "audio", {"confidence": f"{confidence:.3f}"})
            for code in violations
        ]

    @staticmethod
    def _check_secondary_cam(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        """Evaluate SecondaryCameraService output (stub)."""
        violations: List[str] = result.get("violations", [])
        return [
            (code, "webcam_secondary", {})
            for code in violations
        ]

    @staticmethod
    def _check_screen(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        """Evaluate ScreenMonitorService output (stub)."""
        violations: List[str] = result.get("violations", [])
        return [
            (code, "screen", {})
            for code in violations
        ]

    @staticmethod
    def _check_browser(
        result: Dict[str, Any],
    ) -> List[Tuple[str, str, Dict[str, str]]]:
        """Evaluate BrowserMonitorService output (stub)."""
        violations: List[str] = result.get("violations", [])
        return [
            (code, "browser", {})
            for code in violations
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_exclusions(codes: List[str]) -> List[str]:
        """
        Remove violations that are subsumed by a more severe co-occurring one.

        Current exclusion rules:
          - NO_FACE supersedes all LOOKING_AWAY variants (same root cause).
          - GAZE_ESCALATION supersedes GAZE_SUSTAINED_AWAY which supersedes
            LOOKING_AWAY when all three somehow appear (defensive).

        Returns a deduplicated list preserving first-seen order.
        """
        code_set = set(codes)

        excluded: set[str] = set()

        if "NO_FACE" in code_set:
            excluded.update({"LOOKING_AWAY", "GAZE_SUSTAINED_AWAY", "GAZE_ESCALATION"})

        if "GAZE_ESCALATION" in code_set:
            excluded.update({"LOOKING_AWAY", "GAZE_SUSTAINED_AWAY"})
        elif "GAZE_SUSTAINED_AWAY" in code_set:
            excluded.add("LOOKING_AWAY")

        seen: set[str] = set()
        result: List[str] = []
        for code in codes:
            if code not in excluded and code not in seen:
                seen.add(code)
                result.append(code)
        return result

    def _update_frequency(self, active_codes: List[str], now: float) -> None:
        """
        Push current timestamp into each active violation's deque and prune
        entries older than _FREQUENCY_WINDOW_SECONDS.
        """
        cutoff = now - _FREQUENCY_WINDOW_SECONDS
        for code in active_codes:
            self._frequency[code].append(now)
        # Prune all known deques regardless of whether they fired this tick,
        # so counts stay accurate.
        for code in list(self._frequency.keys()):
            dq = self._frequency[code]
            while dq and dq[0] < cutoff:
                dq.popleft()

    def _rolling_count(self, code: str, now: float) -> int:
        """Number of times *code* fired within the last FREQUENCY_WINDOW seconds."""
        cutoff = now - _FREQUENCY_WINDOW_SECONDS
        return sum(1 for t in self._frequency.get(code, deque()) if t >= cutoff)

    def _update_duration(self, active_codes: List[str], now: float) -> None:
        """
        Track start time and last-seen time for each active violation.
        Clear entries for violations that are no longer active.
        """
        active_set = set(active_codes)

        # Update active violations
        for code in active_codes:
            if code in self._active_since:
                start, _ = self._active_since[code]
                self._active_since[code] = (start, now)
            else:
                self._active_since[code] = (now, now)

        # Remove violations that are no longer active
        for code in list(self._active_since.keys()):
            if code not in active_set:
                del self._active_since[code]

    def _aggregate_instant_risk(self) -> float:
        """
        Combine per-violation instant-risk buckets into a single 0–100 value.

        Strategy: non-linear combination so that many small violations cannot
        trivially add up to 100, but a single CRITICAL violation can.

        Formula: 100 * (1 - ∏(1 - r_i / 100))
        This is the "at least one fires" probability model — borrowed from
        reliability engineering.  It ensures the result stays ≤ 100 and grows
        sub-linearly with the number of simultaneous violations.
        """
        product = 1.0
        for v in self._instant_risk_by_violation.values():
            if v > 0.0:
                product *= (1.0 - v / 100.0)
        return round(min((1.0 - product) * 100.0, 100.0), 2)

    @staticmethod
    def _compute_warning_level(session_risk: float) -> WarningLevel:
        if session_risk >= _WARNING_THRESHOLDS[WarningLevel.SEVERE]:
            return WarningLevel.SEVERE
        if session_risk >= _WARNING_THRESHOLDS[WarningLevel.WARNING]:
            return WarningLevel.WARNING
        if session_risk >= _WARNING_THRESHOLDS[WarningLevel.ADVISORY]:
            return WarningLevel.ADVISORY
        return WarningLevel.NONE

    def _check_incidents(
        self,
        active_codes: List[str],
        instant_risk: float,
        session_risk: float,
        now: float,
    ) -> List[IncidentEvent]:
        """
        Raise an IncidentEvent the first time a violation's rolling-window
        count crosses its incident_threshold.

        We track the last count at which we raised an incident and only
        re-raise if the count has grown by at least incident_threshold more
        (so repeated incidents are spaced out, not continuous).
        """
        incidents: List[IncidentEvent] = []

        for code in active_codes:
            rule = get_rule(code)
            if rule is None:
                continue

            count = self._rolling_count(code, now)
            last_raised = self._incident_raised_at[code]
            threshold   = rule.incident_threshold

            if threshold > 0 and count >= last_raised + threshold:
                self._incident_raised_at[code] = count
                inc = IncidentEvent(
                    violation_code = code,
                    severity       = rule.severity,
                    occurrences    = count,
                    session_risk   = session_risk,
                    instant_risk   = instant_risk,
                    timestamp      = now,
                )
                incidents.append(inc)
                logger.warning(
                    "INCIDENT raised  code=%s  severity=%s  occurrences=%d  "
                    "session_risk=%.1f",
                    code, rule.severity.value, count, session_risk,
                )

        return incidents

    @staticmethod
    def _source_for(code: str) -> str:
        rule = get_rule(code)
        if rule and rule.sources:
            return rule.sources[0]
        return "webcam"

    # ------------------------------------------------------------------ #
    # Session reset                                                        #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Clear all stateful tracking.  Call when a new exam session begins
        without recreating the engine object.
        """
        self._instant_risk_by_violation.clear()
        self._session_risk = 0.0
        self._warning_level = WarningLevel.NONE
        self._frequency.clear()
        self._active_since.clear()
        self._incident_raised_at.clear()
        self._last_tick = time.monotonic()
        logger.info("ViolationEngine reset")

    # ------------------------------------------------------------------ #
    # Read-only properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def session_risk(self) -> float:
        """Current EMA session risk (0–100)."""
        return round(self._session_risk, 2)

    @property
    def warning_level(self) -> WarningLevel:
        """Current warning level."""
        return self._warning_level