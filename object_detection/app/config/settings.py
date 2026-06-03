from dataclasses import dataclass
@dataclass
class Settings:
    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    MODEL_PATH = "weights/yolov8m.pt"
    TRACKER_CONFIG = "bytetrack.yaml"
    IMAGE_SIZE = 640
    CONFIDENCE_THRESHOLD = 0.1
    # ---------------------------------------------------
    # DETECTION CLASSES
    # ---------------------------------------------------
    TARGET_CLASSES = [
        0,   # person
        63,  # laptop
        67   # phone
    ]
    # ---------------------------------------------------
    # PERFORMANCE
    # ---------------------------------------------------
    FRAME_SKIP = 10
    MAX_QUEUE_SIZE = 5
    # ---------------------------------------------------
    # CAMERA
    # ---------------------------------------------------
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
settings = Settings()
