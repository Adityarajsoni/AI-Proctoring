

TARGET_FPS = 2

# MediaPipe
MAX_NUM_FACES        = 1
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE  = 0.5
REFINE_LANDMARKS     = True

#Calibration 
# Number of frames collected at exam start while student looks at screen.
# At 2 FPS = 5 frames = ~2.5 seconds.
# During this time the student must look straight at the camera/screen.
CALIBRATION_FRAMES = 10   # ~5 seconds at 2 FPS

# Head pose thresholds (degrees) 
# These are RELATIVE to the student's calibrated neutral position.
# So if their neutral is yaw=-5°, turning right by 35° = yaw_adj=+30°.
YAW_THRESHOLD_DEG = 30    # left/right head turn
PITCH_DOWN_DEG    = 20    # head tilt down (combined with iris check)
PITCH_UP_DEG      = 20    # head tilt up

#Iris vertical (down detection) 
# iris_vertical: 0.0=top(looking up), 0.5=center(screen), 1.0=bottom(down)
# LOOKING_DOWN requires BOTH pitch > PITCH_DOWN_DEG AND iris_v > this
IRIS_DOWN_THRESHOLD = 0.35   # your iris_v when looking down is 0.35-0.39, must be below that

# Pupil-only detection 
PUPIL_DETECTION_ENABLED = True
PUPIL_HEAD_MAX_YAW      = 12   # head must be within ±12° of neutral
PUPIL_HEAD_MAX_PITCH    = 12

# iris_horizontal: 0=left, 0.5=center, 1=right
# Make these wider to reduce false positives (normal eye movement)
PUPIL_LEFT_THRESHOLD  = 0.28   # iris < 0.28 → sustained left gaze
PUPIL_RIGHT_THRESHOLD = 0.72   # iris > 0.72 → sustained right gaze

# Attention state names 
STATE_FOCUSED       = "focused"
STATE_LOOKING_LEFT  = "looking_left"
STATE_LOOKING_RIGHT = "looking_right"
STATE_LOOKING_DOWN  = "looking_down"
STATE_LOOKING_UP    = "looking_up"
STATE_PUPIL_LEFT    = "pupil_left"
STATE_PUPIL_RIGHT   = "pupil_right"
STATE_NO_FACE       = "no_face"
STATE_CALIBRATING   = "calibrating"

# Sustained thresholds 
CONSECUTIVE_AWAY_FRAMES  = 4    # ~2 sec at 2 FPS before sustained event
NO_FACE_SUSTAINED_FRAMES = 6    # ~3 sec of missing face before event
ESCALATION_COUNT         = 3
ESCALATION_WINDOW_FRAMES = 60   # ~30 sec

# Pupil deviation must persist N frames before emitting event
# (prevents single-frame eye blinks/glances triggering events)
PUPIL_SUSTAINED_FRAMES = 3      # ~1.5 sec of sustained pupil shift

#  Risk scores 
#Tune with Risk Engine teammate. Weight = attention_risk × 0.10
RISK_SCORE_FOCUSED        = 0.0
RISK_SCORE_SINGLE_AWAY    = 0.2
RISK_SCORE_SUSTAINED_AWAY = 0.5
RISK_SCORE_REPEATED_AWAY  = 0.8
RISK_SCORE_NO_FACE        = 0.3
RISK_SCORE_PUPIL_AWAY     = 0.4