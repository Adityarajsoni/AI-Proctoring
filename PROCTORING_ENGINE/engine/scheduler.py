# engine/scheduler.py
"""
Frame-rate-aware scheduler.

Tracks a global frame counter and tells callers whether a given
service should run on the current frame.

Design
──────
Pure counter — no threads, no timers, no I/O.
Call tick() once per frame (in ProctoringEngine.process_frame).
Call should_run(name) as many times as needed within that frame.

Default schedule (30 FPS camera):
    gaze          every 1  frame  → ~30 Hz
    anti_spoof    every 5  frames → ~ 6 Hz
    object        every 5  frames → ~ 6 Hz
    verification  every 300 frames → ~10 s
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ── Default schedule ─────────────────────────────────────────────────────────
DEFAULT_SCHEDULE: Dict[str, int] = {
    "gaze":         1,
    "anti_spoof":   5,
    "object":       5,
    "verification": 20,
}
# ─────────────────────────────────────────────────────────────────────────────


class Scheduler:
    """
    Manages per-service frame intervals.

    Usage
    ─────
        scheduler = Scheduler()

        # Once per frame, before calling services:
        scheduler.tick()

        if scheduler.should_run("gaze"):
            result = gaze_tracker.track(frame)

        if scheduler.should_run("anti_spoof"):
            result = anti_spoof.predict(face_crop)
    """

    def __init__(self, schedule: Dict[str, int] | None = None) -> None:
        """
        Parameters
        ----------
        schedule : dict | None
            Maps service name → run every N frames.
            Defaults to DEFAULT_SCHEDULE if not provided.
        """
        self._schedule: Dict[str, int] = schedule or dict(DEFAULT_SCHEDULE)
        self._frame: int = 0

        logger.info("Scheduler initialised  schedule=%s", self._schedule)

    def tick(self) -> int:
        """
        Advance the frame counter by one.
        Call exactly once per incoming camera frame.
        Returns the current frame index (1-based).
        """
        self._frame += 1
        return self._frame

    def should_run(self, service_name: str) -> bool:
        """
        Returns True if *service_name* should execute on the current frame.

        Parameters
        ----------
        service_name : str
            Must be one of the keys in the schedule dict.
            Unknown names always return False and log a warning.
        """
        interval = self._schedule.get(service_name)
        if interval is None:
            logger.warning("Scheduler: unknown service '%s'", service_name)
            return False
        return self._frame % interval == 0

    def reset(self) -> None:
        """Reset frame counter (call between exam sessions)."""
        self._frame = 0
        logger.info("Scheduler reset")

    @property
    def frame_index(self) -> int:
        return self._frame

    def set_interval(self, service_name: str, every_n_frames: int) -> None:
        """Dynamically update the interval for a service at runtime."""
        self._schedule[service_name] = every_n_frames
        logger.info("Scheduler: '%s' interval updated to %d", service_name, every_n_frames)