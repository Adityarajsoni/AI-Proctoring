# test_webcam.py
"""Webcam demo — classifies all faces in frame simultaneously."""

import cv2

from services.anti_spoof_service import AntiSpoofService
from utils.face_cropper import get_all_face_crops

anti_spoof = AntiSpoofService("models/MiniFASNetV2.pth", model_name="v2")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Running — press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_crops = get_all_face_crops(frame)

    for face_patch, bbox in face_crops:
        result = anti_spoof.predict(face_patch)

        x1, y1, x2, y2 = bbox

        if result["label"] == "LIVE":
            color = (0, 255, 0)
        elif result["label"] == "SUSPECT":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f'{result["label"]} {result["confidence"]:.3f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    cv2.imshow("Anti-Spoofing", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()