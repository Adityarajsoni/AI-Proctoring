from ..detectors.yolo_detector import YOLODetector
from ..config.settings import Settings
from ..utils.logger import (
    app_logger,
    error_logger,
    violation_logger,
    performance_logger
)
from ..services.opencv_frame_capture import FrameCapture
from ..services.alert_service import Alert
import cv2
import threading

class SideFaceDetector:
    def __init__(self):
        try:
            self.detector = YOLODetector()
            self.frameCapture = FrameCapture()
            self.frameCapture.start()

            self._stop_event = threading.Event()  # controls the detection loop

            # For alerts
            self.laptop_continuous_frames = 0
            self.laptop_detection_confidence = 0
            self.laptop_count_in_window = []

            self.phone_continuous_frames = 0
            self.phone_detection_confidence = 0

            self.no_person_continuous_frames = 0
            self.no_person_confidence = 1

            self.more_than_one_person_continuous_frames = 0
            self.more_than_one_person_confidence = 0

        except Exception as e:
            error_logger.error(e)

    def start(self):
        app_logger.info("Initializing Side Face Detector")
        self._stop_event.clear()  # ensure flag is cleared before starting

        while not self._stop_event.is_set():
            frame = self.frameCapture.get_frame()

            if frame is None:  # guard: skip if no frame available
                continue

            result = self.detector.detect(frame)
            cv2.imshow("AI Processing - Side Face", result["annotated_frame"])
            cv2.waitKey(1)
            self.process_detections(result)

        self._cleanup()  # runs only after loop exits

    def stop(self):
        app_logger.info("Stopping Side Face Detector")
        self._stop_event.set()  # signals the while loop to exit on next iteration

    def _cleanup(self):
        app_logger.info("Cleaning up Side Face Detector")
        self.frameCapture.stop()
        cv2.destroyAllWindows()

        # reset all state
        self.laptop_continuous_frames = 0
        self.laptop_detection_confidence = 0
        self.laptop_count_in_window = []
        self.phone_continuous_frames = 0
        self.phone_detection_confidence = 0
        self.no_person_continuous_frames = 0
        self.more_than_one_person_continuous_frames = 0
        self.more_than_one_person_confidence = 0

    def process_detections(self, result):
        phone_detected = False
        num_persons_detected = 0
        num_laptops_detected = 0
        person_confidence = []
        laptop_confidences = []

        for detection in result["detections"]:

            if detection["class_id"] == Settings.TARGET_CLASSES[0]:
                num_persons_detected += 1
                person_confidence.append(detection["confidence"])

            if detection["class_id"] == Settings.TARGET_CLASSES[1]:
                num_laptops_detected += 1
                laptop_confidences.append(detection["confidence"])
                violation_logger.info(
                    f"Laptop detected (side face) with confidence {detection['confidence']}"
                )

            if detection["class_id"] == Settings.TARGET_CLASSES[2]:
                phone_detected = True
                violation_logger.info(
                    f"Phone detected (side face) with confidence {detection['confidence']}"
                )
                self.phone_detection_confidence += detection["confidence"]

        # ── Multiple persons ────────────────────────────────────────────────
        if num_persons_detected > 1:
            self.more_than_one_person_continuous_frames += 1
            person_confidence.sort(reverse=True)
            self.more_than_one_person_confidence += person_confidence[1]

            if self.more_than_one_person_continuous_frames == 10:
                self.more_than_one_person_confidence /= 10
                violation_logger.info(
                    f"Multiple persons detected (side face) with confidence "
                    f"{self.more_than_one_person_confidence}"
                )
                Alert("multiple person", self.more_than_one_person_confidence)
                self.more_than_one_person_continuous_frames = 0
                self.more_than_one_person_confidence = 0
        else:
            self.more_than_one_person_continuous_frames = 0
            self.more_than_one_person_confidence = 0

        # ── No person ───────────────────────────────────────────────────────
        if num_persons_detected == 0:
            self.no_person_continuous_frames += 1

            if self.no_person_continuous_frames == 10:
                self.no_person_continuous_frames = 0
                violation_logger.info(
                    "No person found in side face frame with confidence 1"
                )
                Alert("no person", self.no_person_confidence)
        else:
            self.no_person_continuous_frames = 0

        # ── More than one laptop ────────────────────────────────────────────
        if num_laptops_detected > 1:
            self.laptop_continuous_frames += 1
            laptop_confidences.sort(reverse=True)
            self.laptop_detection_confidence += laptop_confidences[1]
            self.laptop_count_in_window.append(num_laptops_detected)

            if self.laptop_continuous_frames == 10:
                avg_confidence = self.laptop_detection_confidence / 10
                violation_logger.info(
                    f"More than one laptop detected (side face) with confidence "
                    f"{avg_confidence}"
                )
                Alert("multiple laptop", avg_confidence)
                self.laptop_continuous_frames = 0
                self.laptop_detection_confidence = 0
                self.laptop_count_in_window = []
        else:
            self.laptop_continuous_frames = 0
            self.laptop_detection_confidence = 0
            self.laptop_count_in_window = []

        # ── Phone ───────────────────────────────────────────────────────────
        if phone_detected:
            self.phone_continuous_frames += 1

            if self.phone_continuous_frames == 10:
                self.phone_detection_confidence /= 10
                violation_logger.info(
                    f"Phone detected (side face) with confidence "
                    f"{self.phone_detection_confidence}"
                )
                Alert("phone", self.phone_detection_confidence)
                self.phone_continuous_frames = 0
                self.phone_detection_confidence = 0
        else:
            self.phone_continuous_frames = 0
            self.phone_detection_confidence = 0