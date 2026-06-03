# test_webcam.py
"""
Full-system integration demo.

Captures webcam frames, passes them through ProctoringEngine,
and prints a live HUD to the console + overlays on the video window.

Usage
─────
    python test_webcam.py [STUDENT_ID]

Before running:
    - Place a reference photo at storage/registration/<STUDENT_ID>/front.jpg
    - Ensure all four services and their model weights are in place.
"""

import logging
import sys
import time

import cv2
import numpy as np

import sys
from pathlib import Path

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))

from engine.proctoring_engine import ProctoringEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
STUDENT_ID             = sys.argv[1] if len(sys.argv) > 1 else "STUDENT001"
ANTI_SPOOF_MODEL_PATH  = "../anti_spoofing/models/MiniFASNetV2.pth"
REFERENCE_IMAGE_PATH   = f"../storage/registration/{STUDENT_ID}/front.jpg"
CAMERA_INDEX           = 0
# ─────────────────────────────────────────────────────────────────────────────


def _draw_hud(frame: np.ndarray, result: dict) -> None:
    """Overlay a minimal HUD on *frame* in-place."""
    font   = cv2.FONT_HERSHEY_SIMPLEX
    white  = (255, 255, 255)
    green  = (0, 255, 0)
    red    = (0, 0, 255)
    yellow = (0, 215, 255)

    violations    = result.get("violations", [])
    cumulative    = result.get("cumulative_risk", 0.0)
    gaze_dir      = result.get("gaze", {}).get("gaze_direction", "?")
    verified      = result.get("verification", {}).get("verified", True)
    is_live       = result.get("anti_spoof", {}).get("is_live", True)
    frame_risk    = result.get("frame_risk_score", 0.0)
    frame_idx     = result.get("frame_index", 0)

    status_color = green if not violations else red

    lines = [
        (f"Frame: {frame_idx}",                          white),
        (f"Gaze:  {gaze_dir}",                           white),
        (f"Verified: {'YES' if verified else 'NO'}",     green if verified else red),
        (f"Live:     {'YES' if is_live  else 'NO'}",     green if is_live  else red),
        (f"Frame Risk:  {frame_risk:.1f}",               white),
        (f"Cumul. Risk: {cumulative:.1f}",               yellow if cumulative > 100 else white),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (15, 35 + i * 32), font, 0.7, color, 2)

    # Violations banner
    if violations:
        banner = "  |  ".join(violations)
        cv2.putText(frame, banner, (15, frame.shape[0] - 20),
                    font, 0.6, red, 2)


def main() -> None:
    logger.info("Starting ProctoringEngine demo  student=%s", STUDENT_ID)

    # ── Load engine ───────────────────────────────────────────────────────
    engine = ProctoringEngine(
        student_id=STUDENT_ID,
        anti_spoof_model_path=ANTI_SPOOF_MODEL_PATH,
    )

    # ── Enroll student ────────────────────────────────────────────────────
    ref_image = cv2.imread(REFERENCE_IMAGE_PATH)
    if ref_image is None:
        logger.warning(
            "No reference image at %s — face verification will be skipped.",
            REFERENCE_IMAGE_PATH,
        )
    else:
        enroll_result = engine.enroll(ref_image)
        logger.info("Enrollment: %s", enroll_result)

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.info("Camera opened — press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # ── Engine call ───────────────────────────────────────────────
            result = engine.process_frame(frame)

            # ── Console summary (every 30 frames) ─────────────────────────
            if result["frame_index"] % 30 == 0:
                logger.info(
                    "frame=%d  gaze=%-12s  verified=%s  live=%s  "
                    "violations=%s  cumulative_risk=%.1f",
                    result["frame_index"],
                    result["gaze"].get("gaze_direction", "?"),
                    result["verification"].get("verified"),
                    result["anti_spoof"].get("is_live"),
                    result["violations"],
                    result["cumulative_risk"],
                )

            # ── Display ───────────────────────────────────────────────────
            _draw_hud(frame, result)
            cv2.imshow("Proctoring Engine", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

        summary = engine.end_session()
        print("\n" + "=" * 52)
        print("  SESSION SUMMARY")
        print("=" * 52)
        s = summary["session"]
        print(f"  Student          : {s['student_id']}")
        print(f"  Duration         : {s['duration_seconds']} s")
        print(f"  Frames processed : {s['total_frames']}")
        # print("\nSession Summary:")
        # print(s)
        print(f"  Peak instant risk : {s['peak_instant_risk']:.2f}")
        print(f"  Peak session risk : {s['peak_session_risk']:.2f}")
        print(f"  Final session risk: {s['final_session_risk']:.2f}")
        print("=" * 52)

        logger.info("Report: %s", summary)


if __name__ == "__main__":
    main()