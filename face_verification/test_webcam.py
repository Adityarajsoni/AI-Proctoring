# test_webcam.py
"""
Minimal webcam demo for FaceVerificationService.

Demonstrates the intended caller pattern:
    1. enroll()  – once, from a reference image or first clean webcam frame
    2. verify()  – every N frames in the capture loop

This file is intentionally thin.  All session tracking (EMA, alerts,
reports) belongs in IdentityTracker / SessionReport if you need it — the
service itself is stateless beyond the stored embedding.
"""

import logging
import sys
import time

import cv2

from services.face_verification_service import FaceVerificationService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
REFERENCE_IMAGE_PATH = "storage/registration/STUDENT001/front.jpg"
CAMERA_INDEX         = 0
VERIFY_EVERY_N       = 2       # run inference every Nth frame
VERIFY_INTERVAL      = 1.5     # minimum seconds between verify() calls
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Loading FaceVerificationService…")
    svc = FaceVerificationService()

    # ── Enrollment ────────────────────────────────────────────────────
    ref_image = cv2.imread(REFERENCE_IMAGE_PATH)
    if ref_image is None:
        logger.error("Could not read reference image: %s", REFERENCE_IMAGE_PATH)
        sys.exit(1)

    enroll_result = svc.enroll(ref_image)
    if not enroll_result["success"]:
        logger.error("Enrollment failed: %s", enroll_result["message"])
        sys.exit(1)

    logger.info("Enrollment OK — %s", enroll_result["message"])

    # ── Webcam loop ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    frame_count      = 0
    last_verify_time = 0.0
    last_result      = None

    logger.info("Running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.monotonic()

        run_inference = (
            frame_count % VERIFY_EVERY_N == 0
            and now - last_verify_time >= VERIFY_INTERVAL
        )

        if run_inference:
            last_verify_time = now
            last_result = svc.verify(frame)

        # ── Draw HUD ──────────────────────────────────────────────────
        if last_result is not None:
            verified   = last_result["verified"]
            similarity = last_result["similarity"]
            confidence = last_result["confidence"]

            label = "VERIFIED" if verified else "NOT VERIFIED"
            color = (0, 255, 0) if verified else (0, 0, 255)

            cv2.putText(frame, label,                          (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Similarity:  {similarity:.4f}", (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Confidence:  {confidence:.1f}%", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "INITIALISING…", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)

        cv2.imshow("Face Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Session ended")


if __name__ == "__main__":
    main()