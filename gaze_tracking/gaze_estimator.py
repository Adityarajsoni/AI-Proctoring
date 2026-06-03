
#  Uses direct face landmark geometry instead:
#    - Yaw  (left/right): ratio of nose-to-eye distances
#    - Pitch (up/down):   nose tip Y position relative to eye line
#    - Iris ratios:       direct pixel position within eye socket
#
#  This approach is:
#    - Numerically stable (no degenerate solutions)
#    - Consistent across all face angles
#    - Produces values in [-1, +1] range (not arbitrary degrees)
#    - Easy to threshold


import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .models.gaze_model import get_face_mesh
from config import (
    YAW_THRESHOLD_DEG,
    PITCH_DOWN_DEG,
    PITCH_UP_DEG,
    IRIS_DOWN_THRESHOLD,
    PUPIL_DETECTION_ENABLED,
    PUPIL_HEAD_MAX_YAW,
    PUPIL_HEAD_MAX_PITCH,
    PUPIL_LEFT_THRESHOLD,
    PUPIL_RIGHT_THRESHOLD,
    STATE_FOCUSED,
    STATE_LOOKING_LEFT,
    STATE_LOOKING_RIGHT,
    STATE_LOOKING_DOWN,
    STATE_LOOKING_UP,
    STATE_PUPIL_LEFT,
    STATE_PUPIL_RIGHT,
    STATE_NO_FACE,
    CALIBRATION_FRAMES,
)

logger = logging.getLogger(__name__)

# Landmark indices 
# Nose
_NOSE_TIP      = 1
_NOSE_BASE     = 168   # between eyes (glabella)

# Eyes outer corners
_L_EYE_OUTER   = 33    # left eye outer corner  (your left)
_R_EYE_OUTER   = 263   # right eye outer corner (your right)
_L_EYE_INNER   = 133
_R_EYE_INNER   = 362

# Eye top/bottom for vertical iris ratio
_L_EYE_TOP     = 159
_L_EYE_BOTTOM  = 145
_R_EYE_TOP     = 386
_R_EYE_BOTTOM  = 374

# Iris centers (only with refine_landmarks=True)
_L_IRIS        = 468
_R_IRIS        = 473

# Chin and forehead for pitch
_CHIN          = 152
_FOREHEAD      = 10


@dataclass
class GazeResult:
    # Normalized scores (-1 to +1)
    # yaw:    -1=fully left,  0=center,  +1=fully right
    # pitch:  -1=fully up,   0=center,  +1=fully down
    yaw:   float
    pitch: float
    roll:  float = 0.0

    # Adjusted (after calibration subtraction)
    yaw_adj:   float = 0.0
    pitch_adj: float = 0.0

    state: str = STATE_NO_FACE

    iris_vertical:   Optional[float] = None
    iris_horizontal: Optional[float] = None

    face_detected: bool = True
    landmarks_2d:  Optional[np.ndarray] = None


# Calibrator 

class GazeCalibrator:
    """Captures neutral pose baseline so deviations are measured correctly."""

    def __init__(self):
        self._yaw_s:   list = []
        self._pitch_s: list = []
        self.calibrated    = False
        self.neutral_yaw   = 0.0
        self.neutral_pitch = 0.0

    def add_sample(self, yaw: float, pitch: float):
        if self.calibrated:
            return
        self._yaw_s.append(yaw)
        self._pitch_s.append(pitch)
        if len(self._yaw_s) >= CALIBRATION_FRAMES:
            self.neutral_yaw   = float(np.median(self._yaw_s))
            self.neutral_pitch = float(np.median(self._pitch_s))
            self.calibrated    = True
            logger.info("Calibration done: neutral_yaw=%.3f  neutral_pitch=%.3f",
                        self.neutral_yaw, self.neutral_pitch)

    def adjust(self, yaw, pitch):
        if not self.calibrated:
            return yaw, pitch
        return yaw - self.neutral_yaw, pitch - self.neutral_pitch

    @property
    def progress(self):
        return f"{len(self._yaw_s)}/{CALIBRATION_FRAMES}"


# Main estimation function 

def estimate_gaze(frame_bgr: np.ndarray,
                  calibrator: Optional[GazeCalibrator] = None) -> GazeResult:
    """
    Estimate gaze from full BGR camera frame.
    Returns GazeResult. Never raises.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return _no_face()

    #  Quick skin-tone check 
    # If no skin-colored pixels exist in the frame, there's no face.
    # This catches "completely empty frame" cases instantly without
    # running MediaPipe at all.
    if not _has_skin(frame_bgr):
        return _no_face()

    h, w = frame_bgr.shape[:2]

    def _run_mediapipe(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        r = get_face_mesh().process(rgb)
        rgb.flags.writeable = True
        return r

    # First attempt — normal frame
    results = _run_mediapipe(frame_bgr)

    # Retry with contrast-boosted frame if face not found
    # MediaPipe struggles with overexposed / dark frames at low confidence
    if not results.multi_face_landmarks:
        boosted = cv2.convertScaleAbs(frame_bgr, alpha=1.3, beta=20)
        results = _run_mediapipe(boosted)

    if not results.multi_face_landmarks:
        return _no_face()

    lms = results.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)

    # Compute yaw from nose-to-eye asymmetry 
    # When head turns RIGHT: nose moves toward LEFT eye
    #   → dist(nose, left_eye) < dist(nose, right_eye)
    # Ratio = (right_dist - left_dist) / (right_dist + left_dist)
    # Range: -1 (fully left) to +1 (fully right)
    nose   = pts[_NOSE_TIP]
    l_eye  = pts[_L_EYE_OUTER]
    r_eye  = pts[_R_EYE_OUTER]

    dist_l = np.linalg.norm(nose - l_eye)
    dist_r = np.linalg.norm(nose - r_eye)
    total  = dist_l + dist_r
    yaw    = (dist_r - dist_l) / total if total > 0 else 0.0

    #Compute pitch from nose vertical position 
    # Nose tip Y relative to the midpoint between forehead and chin
    # When head tilts DOWN: nose tip moves below midpoint → positive pitch
    # When head tilts UP:   nose tip moves above midpoint → negative pitch
    chin     = pts[_CHIN]
    forehead = pts[_FOREHEAD]
    face_h   = chin[1] - forehead[1]            # vertical face span in pixels

    if face_h > 10:
        mid_y  = (chin[1] + forehead[1]) / 2.0
        # Normalize: 0=centered, positive=nose below mid (head down)
        pitch  = (nose[1] - mid_y) / (face_h / 2.0)
        pitch  = float(np.clip(pitch, -1.0, 1.0))
    else:
        pitch = 0.0

    # Calibration 
    yaw_adj, pitch_adj = yaw, pitch
    if calibrator is not None:
        if not calibrator.calibrated:
            calibrator.add_sample(yaw, pitch)
        yaw_adj, pitch_adj = calibrator.adjust(yaw, pitch)

    # Iris ratios 
    has_iris = len(pts) >= 478
    iris_v   = _iris_vertical(pts)   if has_iris else None
    iris_h   = _iris_horizontal(pts) if has_iris else None

    # Classify 
    state = _classify(yaw_adj, pitch_adj, iris_v, iris_h)

    logger.info(
        "state=%-14s  yaw=%+.3f(adj %+.3f)  pitch=%+.3f(adj %+.3f)  iv=%-4s  ih=%-4s",
        state, yaw, yaw_adj, pitch, pitch_adj,
        f"{iris_v:.2f}" if iris_v is not None else "N/A",
        f"{iris_h:.2f}" if iris_h is not None else "N/A",
    )

    return GazeResult(
        yaw=yaw, pitch=pitch, roll=0.0,
        yaw_adj=yaw_adj, pitch_adj=pitch_adj,
        state=state,
        iris_vertical=iris_v,
        iris_horizontal=iris_h,
        face_detected=True,
        landmarks_2d=pts,
    )


# Classification 
# Thresholds are now RATIOS (0.0 to 1.0), not degrees.
# Update config.py thresholds accordingly (see below).

def _classify(yaw: float, pitch: float,
              iris_v: Optional[float],
              iris_h: Optional[float]) -> str:

    yaw_thresh     = YAW_THRESHOLD_DEG / 180.0   # ~0.17 for 30°
    pitch_d_thresh = PITCH_DOWN_DEG    / 90.0    # ~0.22 for 20°
    pitch_u_thresh = PITCH_UP_DEG      / 90.0

    # Strong pitch always wins — if head is clearly tilted up or down,
    # classify that first before checking yaw.
    # This prevents "looking down" being swallowed by "looking_left"
    # when someone tilts their head while turning slightly.
    STRONG_PITCH = pitch_d_thresh * 1.5   # ~0.33 — unambiguous tilt

    # 1. Strong downward pitch — no iris check needed (iris unreliable at extremes)
    if pitch > STRONG_PITCH:
        return STATE_LOOKING_DOWN

    # 2. Strong upward pitch
    if pitch < -STRONG_PITCH:
        return STATE_LOOKING_UP

    # 3. Head left / right (only checked when pitch is not dominant)
    if yaw > yaw_thresh:
        return STATE_LOOKING_RIGHT
    if yaw < -yaw_thresh:
        return STATE_LOOKING_LEFT

    # 4. Moderate down pitch + iris confirmation (keyboard-aware)
    if pitch > pitch_d_thresh:
        if iris_v is None or iris_v > IRIS_DOWN_THRESHOLD:
            return STATE_LOOKING_DOWN
        return STATE_FOCUSED   # head tilted but iris on screen = typing

    # 5. Moderate up pitch
    if pitch < -pitch_u_thresh:
        return STATE_LOOKING_UP

    # 6. Pupil-only (head still, eyes shift sideways)
    if (PUPIL_DETECTION_ENABLED and iris_h is not None
            and abs(yaw)   < (PUPIL_HEAD_MAX_YAW   / 180.0)
            and abs(pitch) < (PUPIL_HEAD_MAX_PITCH  / 90.0)):
        if iris_h < PUPIL_LEFT_THRESHOLD:
            return STATE_PUPIL_LEFT
        if iris_h > PUPIL_RIGHT_THRESHOLD:
            return STATE_PUPIL_RIGHT

    return STATE_FOCUSED


# Iris helpers 

def _iris_vertical(pts: np.ndarray) -> Optional[float]:
    try:
        def r(top_i, bot_i, iris_i):
            t = min(pts[top_i][1], pts[bot_i][1])
            b = max(pts[top_i][1], pts[bot_i][1])
            span = b - t
            return float(np.clip((pts[iris_i][1] - t) / span, 0, 1)) if span > 2 else 0.5
        return (r(_L_EYE_TOP, _L_EYE_BOTTOM, _L_IRIS) +
                r(_R_EYE_TOP, _R_EYE_BOTTOM, _R_IRIS)) / 2
    except (IndexError, ZeroDivisionError):
        return None


def _iris_horizontal(pts: np.ndarray) -> Optional[float]:
    try:
        def r(outer_i, inner_i, iris_i):
            left  = min(pts[outer_i][0], pts[inner_i][0])
            right = max(pts[outer_i][0], pts[inner_i][0])
            span  = right - left
            return float(np.clip((pts[iris_i][0] - left) / span, 0, 1)) if span > 2 else 0.5
        return (r(_L_EYE_OUTER, _L_EYE_INNER, _L_IRIS) +
                r(_R_EYE_OUTER, _R_EYE_INNER, _R_IRIS)) / 2
    except (IndexError, ZeroDivisionError):
        return None


def _has_skin(frame_bgr: np.ndarray, min_skin_ratio: float = 0.01) -> bool:
    """
    Returns True if the frame contains enough skin-colored pixels
    to plausibly contain a face.

    Uses HSV skin range — works across different skin tones.
    min_skin_ratio=0.01 means at least 1% of frame must be skin.
    At 640x480 that's ~3000 pixels — enough for even a distant face.
    """
    # Downsample for speed (check 1/4 resolution)
    small = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # Skin tone range in HSV — covers light to dark skin
    lower = np.array([0,  20,  50], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    skin_ratio = np.count_nonzero(mask) / mask.size
    return skin_ratio >= min_skin_ratio


def _no_face() -> GazeResult:
    return GazeResult(yaw=0, pitch=0, roll=0,
                      state=STATE_NO_FACE, face_detected=False)



