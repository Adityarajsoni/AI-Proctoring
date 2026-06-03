# violation/violation_rules.py
"""
Violation rule definitions.

Design
──────
Each rule carries:
  - severity    : LOW | MEDIUM | HIGH | CRITICAL
  - base_weight : raw impact used by the risk model (0–100 scale anchor)
  - decay_rate  : fraction subtracted per second of clean frames (0.0–1.0)
  - escalation_threshold : times seen in rolling window before upgrading severity
  - incident_threshold   : cumulative occurrences before opening an Incident

Tuning guide
────────────
  base_weight    governs instant-risk spike magnitude.
  decay_rate     governs how quickly instant risk falls back after the
                 violation stops.  Higher = faster recovery.
  escalation_threshold  how many times the violation fires (in the rolling
                 30-second window) before it is treated as the next severity
                 tier.  Set to 0 to disable escalation for that rule.
  incident_threshold    total session occurrences before the violation is
                 promoted to an Incident record.

Severity tiers (used by SessionManager for warning levels):
  LOW      → advisory, no warning issued
  MEDIUM   → WARNING level 1
  HIGH     → WARNING level 2 (second warning)
  CRITICAL → immediate INCIDENT, exam may be flagged for review

Future monitors (audio, dual-camera, screen, browser) add new entries here
only — no other file needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


# ── Severity ──────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def warning_level(self) -> int:
        """Map severity to a 0-based warning level integer."""
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[self.value]


# ── Rule dataclass ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViolationRule:
    name:                  str
    severity:              Severity
    base_weight:           float          # 0–100; used as instant-risk spike
    decay_rate:            float          # fraction per second, e.g. 0.15 → -15% /s
    description:           str
    escalation_threshold:  int   = 3      # rolling-window hits before escalation
    incident_threshold:    int   = 5      # session total before Incident is raised
    # Which monitor sources produce this violation (for future fusion).
    # Empty tuple means "any / legacy".
    sources:               tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not (0.0 < self.base_weight <= 100.0):
            raise ValueError(
                f"Rule '{self.name}': base_weight must be in (0, 100], "
                f"got {self.base_weight}"
            )
        if not (0.0 < self.decay_rate <= 1.0):
            raise ValueError(
                f"Rule '{self.name}': decay_rate must be in (0, 1], "
                f"got {self.decay_rate}"
            )


# ── Rule registry ─────────────────────────────────────────────────────────────
#
# base_weight design rationale
# ─────────────────────────────
# We treat 100.0 as "maximally certain cheating".  Grades:
#   CRITICAL  75–100   (spoof, identity swap, extra person)
#   HIGH      40–74    (phone, gaze pattern, multiple laptops)
#   MEDIUM    15–39    (no person briefly, one laptop, gaze sustained)
#   LOW        1–14    (single brief gaze-away, no face for 1 frame)
#
# decay_rate design rationale
# ────────────────────────────
# Fast-decay (0.25–0.40): transient violations like LOOKING_AWAY that
#   resolve themselves within a few seconds.
# Slow-decay (0.05–0.10): structural violations like SPOOF_DETECTED or
#   FACE_MISMATCH that indicate persistent intent to cheat and should
#   stay elevated until a human reviewer clears them.

RULES: Dict[str, ViolationRule] = {

    # ── Identity / integrity ───────────────────────────────────────────────
    "SPOOF_DETECTED": ViolationRule(
        name                 = "SPOOF_DETECTED",
        severity             = Severity.CRITICAL,
        base_weight          = 100.0,
        decay_rate           = 0.05,   # stays high; operator must clear
        escalation_threshold = 1,      # one occurrence already escalates
        incident_threshold   = 1,
        description          = "Anti-spoofing check failed — possible photo/video attack",
        sources              = ("webcam",),
    ),
    "FACE_MISMATCH": ViolationRule(
        name                 = "FACE_MISMATCH",
        severity             = Severity.CRITICAL,
        base_weight          = 100.0,
        decay_rate           = 0.05,
        escalation_threshold = 1,
        incident_threshold   = 1,
        description          = "Face verification failed — identity mismatch",
        sources              = ("webcam",),
    ),

    # ── People in frame ────────────────────────────────────────────────────
    "MULTIPLE_PERSONS": ViolationRule(
        name                 = "MULTIPLE_PERSONS",
        severity             = Severity.CRITICAL,
        base_weight          = 90.0,
        decay_rate           = 0.10,
        escalation_threshold = 2,
        incident_threshold   = 2,
        description          = "More than one person detected in frame",
        sources              = ("webcam",),
    ),
    "NO_PERSON": ViolationRule(
        name                 = "NO_PERSON",
        severity             = Severity.MEDIUM,
        base_weight          = 30.0,
        decay_rate           = 0.30,   # recovers quickly when student returns
        escalation_threshold = 5,
        incident_threshold   = 10,
        description          = "No person detected in frame",
        sources              = ("webcam",),
    ),

    # ── Prohibited objects ─────────────────────────────────────────────────
    "PHONE_DETECTED": ViolationRule(
        name                 = "PHONE_DETECTED",
        severity             = Severity.HIGH,
        base_weight          = 70.0,
        decay_rate           = 0.15,
        escalation_threshold = 3,
        # Re-raise an incident every 10 rolling-window occurrences.
        # A phone sitting in frame for an entire 60-second window will fire
        # once (at 10 hits) rather than on every single frame.
        incident_threshold   = 10,
        description          = "Mobile phone visible in frame",
        sources              = ("webcam",),
    ),
    "LAPTOP_DETECTED": ViolationRule(
        name                 = "LAPTOP_DETECTED",
        severity             = Severity.MEDIUM,
        base_weight          = 25.0,
        decay_rate           = 0.20,
        escalation_threshold = 5,
        incident_threshold   = 8,
        description          = "Secondary laptop detected in frame",
        sources              = ("webcam",),
    ),
    "MULTIPLE_LAPTOPS": ViolationRule(
        name                 = "MULTIPLE_LAPTOPS",
        severity             = Severity.HIGH,
        base_weight          = 55.0,
        decay_rate           = 0.15,
        escalation_threshold = 3,
        incident_threshold   = 3,
        description          = "Multiple laptops detected",
        sources              = ("webcam",),
    ),

    # ── Gaze ──────────────────────────────────────────────────────────────
    "LOOKING_AWAY": ViolationRule(
        name                 = "LOOKING_AWAY",
        severity             = Severity.LOW,
        base_weight          = 12.0,
        decay_rate           = 0.40,   # single glance decays very fast
        escalation_threshold = 6,
        # 30 firings in the 60-second window before re-raising an incident.
        # A student glancing away every 2 seconds for a full minute triggers
        # one incident; occasional glances do not.
        incident_threshold   = 30,
        description          = "Student gaze briefly directed away from screen",
        sources              = ("webcam",),
    ),
    "GAZE_SUSTAINED_AWAY": ViolationRule(
        name                 = "GAZE_SUSTAINED_AWAY",
        severity             = Severity.MEDIUM,
        base_weight          = 35.0,
        decay_rate           = 0.25,
        escalation_threshold = 4,
        incident_threshold   = 8,
        description          = "Student sustained gaze away for multiple frames",
        sources              = ("webcam",),
    ),
    "GAZE_ESCALATION": ViolationRule(
        name                 = "GAZE_ESCALATION",
        severity             = Severity.HIGH,
        base_weight          = 60.0,
        decay_rate           = 0.12,
        escalation_threshold = 2,
        incident_threshold   = 4,
        description          = "Repeated sustained gaze violations in rolling window",
        sources              = ("webcam",),
    ),
    "NO_FACE": ViolationRule(
        name                 = "NO_FACE",
        severity             = Severity.LOW,
        base_weight          = 15.0,
        decay_rate           = 0.35,
        escalation_threshold = 8,
        incident_threshold   = 15,
        description          = "No face detected by gaze tracker",
        sources              = ("webcam",),
    ),

    # ── Future: Audio monitor (reserved — not yet active) ─────────────────
    # Uncomment when AudioMonitorService is wired in.
    #
    # "VOICE_DETECTED": ViolationRule(
    #     name="VOICE_DETECTED", severity=Severity.MEDIUM, base_weight=30.0,
    #     decay_rate=0.30, escalation_threshold=4, incident_threshold=6,
    #     description="External voice detected during exam",
    #     sources=("audio",),
    # ),
    # "MULTIPLE_VOICES": ViolationRule(
    #     name="MULTIPLE_VOICES", severity=Severity.HIGH, base_weight=65.0,
    #     decay_rate=0.15, escalation_threshold=2, incident_threshold=3,
    #     description="Multiple distinct voices detected",
    #     sources=("audio",),
    # ),

    # ── Future: Dual-camera monitor (reserved) ─────────────────────────────
    # "SECONDARY_DEVICE_VISIBLE": ViolationRule(
    #     name="SECONDARY_DEVICE_VISIBLE", severity=Severity.HIGH,
    #     base_weight=65.0, decay_rate=0.15, escalation_threshold=2,
    #     incident_threshold=3, description="Secondary device visible on rear cam",
    #     sources=("webcam_secondary",),
    # ),

    # ── Future: Screen / browser monitor (reserved) ────────────────────────
    # "PROHIBITED_TAB_OPEN": ViolationRule(
    #     name="PROHIBITED_TAB_OPEN", severity=Severity.HIGH, base_weight=60.0,
    #     decay_rate=0.10, escalation_threshold=2, incident_threshold=3,
    #     description="Prohibited browser tab or window detected",
    #     sources=("screen", "browser"),
    # ),
    # "COPY_PASTE_DETECTED": ViolationRule(
    #     name="COPY_PASTE_DETECTED", severity=Severity.MEDIUM, base_weight=40.0,
    #     decay_rate=0.20, escalation_threshold=3, incident_threshold=5,
    #     description="Clipboard copy/paste operation detected during exam",
    #     sources=("browser",),
    # ),
}


# ── Public helpers ────────────────────────────────────────────────────────────

def get_rule(name: str) -> ViolationRule | None:
    """Return the rule for *name*, or None if not registered."""
    return RULES.get(name)


def get_weight(name: str, default: float = 0.0) -> float:
    """Return the base_weight for *name*, or *default* if not found."""
    rule = RULES.get(name)
    return rule.base_weight if rule else default


def get_severity(name: str) -> Severity:
    """Return the Severity for *name*, defaulting to LOW."""
    rule = RULES.get(name)
    return rule.severity if rule else Severity.LOW


def get_decay_rate(name: str, default: float = 0.20) -> float:
    """Return the per-second decay rate for *name*."""
    rule = RULES.get(name)
    return rule.decay_rate if rule else default