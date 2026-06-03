"""
test_webcam.py
==============
Manual integration test for ObjectDetectionService.

Run from the project root:
    python -m object_detection.test_webcam

Press ESC to quit.
This file is the ONLY place that contains webcam / cv2.imshow code.
"""

import cv2
from object_detection.services import ObjectDetectionService
from app.config.settings import settings


def main() -> None:
    service = ObjectDetectionService()

    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frame_count = 0

    print("ObjectDetectionService test — press ESC to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ── frame skipping (mirrors original CVRunner behaviour) ──────────
        frame_count += 1
        if frame_count % settings.FRAME_SKIP != 0:
            continue

        # ── service call ──────────────────────────────────────────────────
        result = service.detect(frame)

        # ── console output ────────────────────────────────────────────────
        print(
            f"persons={result['person_count']:2d} | "
            f"phone={result['phone_detected']} | "
            f"laptops={result['laptop_count']} | "
            f"violations={result['violations']}"
        )

        # ── minimal display (test only) ───────────────────────────────────
        cv2.imshow("ObjectDetectionService — test", frame)
        if cv2.waitKey(1) == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()