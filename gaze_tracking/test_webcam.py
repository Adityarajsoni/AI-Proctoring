# test_webcam.py
"""
Minimal webcam demo for GazeTrackingService.
No drawing logic, no UI code — console output only.
Runs at TARGET_FPS (2 FPS) matching the design frequency.
"""

import logging
import time

import cv2

from services.gaze_tracking_service import GazeTrackingService
from config import TARGET_FPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

STUDENT_ID   = "STUDENT001"
CAMERA_INDEX = 0


def main() -> None:
    logger.info("Starting GazeTrackingService demo  student=%s", STUDENT_ID)

    svc = GazeTrackingService(student_id=STUDENT_ID)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    frame_interval = 1.0 / TARGET_FPS   # 0.5 s between inference calls
    last_inference = 0.0

    logger.info("Running at %d FPS — press 'q' to quit", TARGET_FPS)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.monotonic()
            if now - last_inference >= frame_interval:
                last_inference = now

                result = svc.track(frame)

                logger.info(
                    "frame=%d  dir=%-12s  yaw_adj=%+.3f  pitch_adj=%+.3f"
                    "  risk=%.2f  attentive=%s  calibrated=%s  progress=%s",
                    result["frame_index"],
                    result["gaze_direction"],
                    result["yaw_adj"],
                    result["pitch_adj"],
                    result["risk_score"],
                    result["is_attentive"],
                    result["calibrated"],
                    result["calibration_progress"],
                )

                if result["event"] is not None:
                    logger.warning("EVENT: %s", result["event"])

            # Still show video at full camera FPS — no imshow logic in service
            cv2.imshow("Gaze Tracking (raw feed)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

        summary = svc.end_session()
        logger.info("Session summary: %s", summary)

        svc.shutdown()


if __name__ == "__main__":
    main()