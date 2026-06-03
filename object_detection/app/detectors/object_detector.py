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

class ObjectDetector:
    def __init__(self):
        try:
            self.detector = YOLODetector()
            self.frameCapture = FrameCapture()
            self.frameCapture.start()

            self._stop_event = threading.Event()
            #for alerts
            self.laptop_continuous_frames = 0
            self.laptop_detection_confidence = 0

            self.phone_continuous_frames = 0
            self.phone_detetion_confidence = 0

            self.no_person_continuous_frames = 0
            self.no_person_confidence = 1

            self.more_than_one_continuous_frames = 0
            self.more_than_one_person_confidence = 0

        except Exception as e:
            error_logger.error(e)

    def start(self):
        app_logger.info("Initializing Object Detector")
        self._stop_event.clear()
        while not self._stop_event.is_set():
            frame = self.frameCapture.get_frame()
            if frame is None:
                continue
            result = self.detector.track(frame)
            cv2.imshow("AI Processing",result["annotated_frame"])
            cv2.waitKey(1)
            self.process_detetions(result)
        self._cleanup()
    
    def stop(self):
        app_logger.info("Stopping Side Face Detector")
        self._stop_event.set()

    def _cleanup(self):
        app_logger.info("Cleaning up Side Face Detector")
        self.frameCapture.stop()
        cv2.destroyAllWindows()

        # reset all state
        self.laptop_continuous_frames = 0
        self.laptop_detection_confidence = 0
        self.phone_continuous_frames = 0
        self.phone_detetion_confidence = 0
        self.no_person_continuous_frames = 0
        self.more_than_one_continuous_frames = 0
        self.more_than_one_person_confidence = 0

    def process_detetions(self, result):
        phone_detected = False
        laptop_detected = False
        num_persons_detected = 0
        person_confidence = []
        for detection in result["detections"]:
                
            #person detection
            if(detection["class_id"] == Settings.TARGET_CLASSES[0]):
                num_persons_detected += 1
                person_confidence.append(detection['confidence'])
                
            #laptop detection
            if(detection["class_id"] == Settings.TARGET_CLASSES[1]):
                laptop_detected = True
                violation_logger.info(f"Laptop detected with confidence {detection["confidence"]}")
                self.laptop_detection_confidence += detection['confidence']
                
            #phone detected
            if(detection["class_id"] == Settings.TARGET_CLASSES[2]):
                phone_detected = True
                violation_logger.info(f"Phone detected with confidence {detection["confidence"]}")
                self.phone_detetion_confidence += detection['confidence']
        
        if(num_persons_detected > 1):
            self.more_than_one_continuous_frames += 1
            person_confidence.sort(reverse=True)
            self.more_than_one_person_confidence += person_confidence[1]
            if(self.more_than_one_continuous_frames == 10):
                self.more_than_one_person_confidence /= 10
                self.more_than_one_continuous_frames = 0
                #alert more than one person detected
                violation_logger.info(f"More than two persons detected with confidence {person_confidence[1]}")
                Alert("multiple person", self.more_than_one_person_confidence)
                self.more_than_one_person_confidence = 0
        else:
            self.more_than_one_continuous_frames = 0
            self.more_than_one_person_confidence = 0

        if(num_persons_detected == 0):
            self.no_person_continuous_frames += 1
            if(self.no_person_continuous_frames == 10):
                self.no_person_continuous_frames = 0
                #alert no person detected
                violation_logger.info("No person found in the detected with confdence 1")
                Alert("no person", self.no_person_confidence)
        else:
            self.no_person_continuous_frames = 0

        if(laptop_detected):
            self.laptop_continuous_frames += 1
            if(self.laptop_continuous_frames == 10):
                self.laptop_continuous_frames = 0
                self.laptop_detection_confidence /= 10
                #alert laptop detected
                # violation_logger.info("Laptop detected with confidence")
                Alert("laptop", self.laptop_detection_confidence)
                self.laptop_detection_confidence = 0
        else:
            self.laptop_continuous_frames = 0
            self.laptop_detection_confidence = 0
        if(phone_detected):
            self.phone_continuous_frames += 1
            if(self.phone_continuous_frames == 10):
                self.phone_continuous_frames = 0
                self.phone_detetion_confidence /= 10
                #alert phone detected
                # violation_logger.info("Phone detected")
                Alert("phone", self.phone_detetion_confidence)
                self.phone_detetion_confidence = 0
        else:
            self.phone_continuous_frames = 0
            self.phone_detetion_confidence = 0
                