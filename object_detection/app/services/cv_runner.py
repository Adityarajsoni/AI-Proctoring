import cv2
import time

from ..detectors.yolo_detector import YOLODetector
from ..config.settings import settings

from ..utils.logger import (
    app_logger,
    violation_logger,
    performance_logger
)


class CVRunner:
    def __init__(self):

        app_logger.info("Initializing CV Runner")
        self.detector = YOLODetector()
        self.detector.warmup()
        self.cap = cv2.VideoCapture(
            settings.CAMERA_INDEX
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            settings.FRAME_WIDTH
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            settings.FRAME_HEIGHT
        )
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )
        if not self.cap.isOpened():
            raise RuntimeError(
                "Failed to initialize webcam"
            )
        self.frame_count = 0
        self.prev_time = time.time()

        #for alerts
        self.laptop_continues_frames = 0
        self.phone_continues_frames = 0
        self.no_person_continues_frames = 0
        self.more_than_one_person_continues_frames = 0
    def start(self):
        app_logger.info("CV Runner started")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                app_logger.warning(
                    "Failed to read frame"
                )
                continue
            # -----------------------------------------
            # FRAME SKIPPING
            # -----------------------------------------
            self.frame_count += 1
            if (
                self.frame_count %
                settings.FRAME_SKIP != 0
            ):
                continue
            # -----------------------------------------
            # DETECTION
            # -----------------------------------------
            result = self.detector.track(frame)
            detections = result["detections"]
            annotated_frame = result["annotated_frame"]
            # -----------------------------------------
            # BUSINESS LOGIC
            # -----------------------------------------
            self.process_detections(detections)
            # -----------------------------------------
            # FPS
            # -----------------------------------------
            self.calculate_fps()
            # -----------------------------------------
            # DISPLAY
            # -----------------------------------------
            cv2.imshow(
                "AI Proctoring",
                annotated_frame
            )
            key = cv2.waitKey(1)
            if key == 27:
                app_logger.info(
                    "ESC pressed"
                )
                break
        self.cleanup()
    def process_detections(self, detections):
        phone_detected = False
        laptop_detected = False
        more_than_two_persons_detected = False
        no_person_detected = False
        num_humans_detected = 0
        for detection in detections:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            # -------------------------------------
            # PHONE DETECTION
            # -------------------------------------
            if class_name == "cell phone":
                phone_detected = True
                violation_logger.warning(
                    f"PHONE DETECTED | "
                    f"CONF={confidence:.2f}"
                )
            # -------------------------------------
            # LAPTOP DETECTION
            # -------------------------------------
            if class_name == "laptop":
                laptop_detected = True
                violation_logger.warning(
                    f"LAPTOP DETECTED | "
                    f"CONF={confidence:.2f}"
                )
            # -------------------------------------
            # HUMAN DETECTION
            # -------------------------------------
            if class_name == "person":
                num_humans_detected += 1
            
            
    def calculate_fps(self):
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed >= 1:
            fps = self.frame_count / elapsed
            performance_logger.info(
                f"FPS={fps:.2f}"
            )
            self.prev_time = current_time
            self.frame_count = 0
    def cleanup(self):
        app_logger.info("Cleaning up")
        self.cap.release()
        cv2.destroyAllWindows()
