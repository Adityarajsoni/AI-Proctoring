import cv2
from ..config.settings import Settings
from ..utils.logger import (
    app_logger,
    error_logger,
    performance_logger
)
import threading

class FrameCapture:
    def __init__(self):
        try:
            self.running = False
            self.frame = None
            self.lock = threading.Lock()
            self._stop_event = threading.Event()

            app_logger.info("Initializing OpenCV video capture")
            self.cap = cv2.VideoCapture(Settings.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Settings.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Settings.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal buffer

        except Exception as e:
            error_logger.error(e)
            raise

    def start(self):
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self._thread.start()
        app_logger.info("Frame capture started")

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def _capture_loop(self):
        frame_count = 0

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret:
                error_logger.error("Failed to read frame from camera")
                continue

            frame_count += 1  # increment FIRST, before any skip check

            # skip frames to reduce detection load
            if frame_count % Settings.FRAME_SKIP != 0:
                continue  # read and discard — drains the buffer properly

            with self.lock:
                self.frame = frame

    def stop(self):
        app_logger.info("Stopping frame capture")
        self._stop_event.set()
        self.running = False

        if hasattr(self, '_thread'):
            self._thread.join(timeout=2)  # wait for loop to exit cleanly

        if self.cap.isOpened():
            self.cap.release()
        app_logger.info("Frame capture stopped")