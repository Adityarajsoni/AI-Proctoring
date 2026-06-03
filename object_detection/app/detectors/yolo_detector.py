from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict, Any

from ..config.settings import settings
from ..utils.logger import (
    app_logger,
    violation_logger,
    performance_logger,
    error_logger
)

class YOLODetector:
    def __init__(self):
        #loading model yolo on half precision
        try:
            app_logger.info("Initializing YOLO detector")
            self.device = self._select_device()
            app_logger.info(f"Using device: {self.device}")
            self.model = YOLO(settings.MODEL_PATH)
            self.model.to(self.device)
            # FP16 optimization
            if self.device == "cuda":
                self.model.model.half()
                performance_logger.info("FP16 enabled")
            app_logger.info("YOLO model loaded successfully")
        #on failure to load the model
        except Exception as e:
            error_logger.exception(f"Failed to initialize YOLO model: {e}")
            raise

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def detect(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:

        """
        Run inference on frame

        Returns:
        {
            "detections": [...],
            "annotated_frame": frame
        }
        """
        try:
            results = self.model.predict(
                source=frame,
                conf=settings.CONFIDENCE_THRESHOLD,
                classes=settings.TARGET_CLASSES,
                imgsz=settings.IMAGE_SIZE,
                device=self.device,
                verbose=False
            )
            result = results[0]
            detections = self._parse_detections(result)
            annotated_frame = result.plot()
            return {
                "detections": detections,
                "annotated_frame": annotated_frame
            }

        except Exception as e:
            error_logger.exception(f"Inference failed: {e}")
            return {
                "detections": [],
                "annotated_frame": frame
            }

    def track(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:

        """
        Run YOLO + ByteTrack
        """
        try:
            results = self.model.track(
                source=frame,
                conf=settings.CONFIDENCE_THRESHOLD,
                classes=settings.TARGET_CLASSES,
                imgsz=settings.IMAGE_SIZE,
                device=self.device,
                tracker=settings.TRACKER_CONFIG,
                persist=True,
                verbose=False
            )
            result = results[0]
            detections = self._parse_tracking(result)
            annotated_frame = result.plot()
            return {
                "detections": detections,
                "annotated_frame": annotated_frame
            }

        except Exception as e:
            error_logger.exception(f"Tracking failed: {e}")
            return {
                "detections": [],
                "annotated_frame": frame
            }

    def _parse_detections(
        self,
        result
    ) -> List[Dict]:

        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )
            detection = {
                "class_id": cls_id,
                "class_name": self.model.names[cls_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(detection)
        return detections

    def _parse_tracking(
        self,
        result
    ) -> List[Dict]:
        detections = []
        if result.boxes is None:
            return detections
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )
            detection = {
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": self.model.names[cls_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(detection)
        return detections

    def warmup(self):
        """
        Warmup model for lower first-inference latency
        """
        app_logger.info("Running model warmup")
        dummy_frame = np.zeros(
            (
                settings.IMAGE_SIZE,
                settings.IMAGE_SIZE,
                3
            ),
            dtype=np.uint8
        )
        self.detect(dummy_frame)
        app_logger.info("Warmup completed")
