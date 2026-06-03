from ..detectors.object_detector import ObjectDetector
from ..config.settings import Settings

class PersonFrame:
    def __init__(self):
        self.detector = ObjectDetector()
    
    def get_person_frame(self):
        result = self.detector.detect()