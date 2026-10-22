"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Configuration File
==========================================================
"""

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Sign Language Recognition System"

PROJECT_TITLE = (
    "Machine Learning Based Real-Time "
    "Sign Language Recognition with Speech Conversion"
)

VERSION = "Version 1.0"

COLLEGE_NAME = "Sarah Tucker College (Autonomous)"

DEPARTMENT = "Department of Computer Applications"

COURSE = "Master of Computer Applications (MCA)"

GUIDE_NAME = "Dr. Evangelin Deva Sheela"

DEVELOPER = "Fairlin Gold Teena"

# ==========================================================
# MODEL PATHS
# ==========================================================

MODEL_PATH = "models/hand_landmarker.task"

GESTURE_MODEL_PATH = "models/gesture_model.pkl"

SCALER_PATH = "models/scaler.pkl"

LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# ==========================================================
# CAMERA SETTINGS
# ==========================================================

CAMERA_ID = 0

FRAME_WIDTH = 640

FRAME_HEIGHT = 480

FPS = 30

# ==========================================================
# MEDIAPIPE SETTINGS
# ==========================================================

NUM_HANDS = 2

MIN_HAND_DETECTION_CONFIDENCE = 0.35

MIN_HAND_PRESENCE_CONFIDENCE = 0.35

MIN_TRACKING_CONFIDENCE = 0.35

# ==========================================================
# PREDICTION SETTINGS
# ==========================================================

STABLE_FRAMES = 8

MIN_CONFIDENCE = 85.0

# ==========================================================
# TEXT TO SPEECH
# ==========================================================

SPEECH_RATE = 150

SPEECH_VOLUME = 1.0

# ==========================================================
# WINDOW SETTINGS
# ==========================================================

WINDOW_TITLE = PROJECT_NAME

WINDOW_WIDTH = 1280

WINDOW_HEIGHT = 700

# ==========================================================
# THEME COLORS
# ==========================================================

BACKGROUND = "#0F172A"

HEADER = "#1E3A8A"

CARD = "#1E293B"

PRIMARY = "#2563EB"

SECONDARY = "#334155"

SUCCESS = "#22C55E"

WARNING = "#F59E0B"

DANGER = "#EF4444"

TEXT = "#FFFFFF"

SUBTEXT = "#CBD5E1"

ACCENT = "#06B6D4"

BORDER = "#3B82F6"

# ==========================================================
# OPENCV COLORS (BGR)
# ==========================================================

LANDMARK_COLOR = (0, 255, 0)

CONNECTION_COLOR = (255, 0, 0)

GESTURE_TEXT_COLOR = (0, 255, 255)

CONFIDENCE_TEXT_COLOR = (0, 255, 0)

WARNING_TEXT_COLOR = (0, 0, 255)

# ==========================================================
# UI FONTS
# ==========================================================

TITLE_FONT = ("Segoe UI", 28, "bold")

SUBTITLE_FONT = ("Segoe UI", 15)

HEADING_FONT = ("Segoe UI", 18, "bold")

NORMAL_FONT = ("Segoe UI", 15)

SMALL_FONT = ("Segoe UI", 12)

BUTTON_FONT = ("Segoe UI", 15, "bold")

# ==========================================================
# PANEL SETTINGS
# ==========================================================

HEADER_HEIGHT = 100

FOOTER_HEIGHT = 70

CAMERA_PANEL_WIDTH = 850

CAMERA_PANEL_HEIGHT = 360

INFO_PANEL_WIDTH = 350

INFO_PANEL_HEIGHT = 360

HISTORY_PANEL_HEIGHT = 95

CONTROL_PANEL_HEIGHT = 65

# ==========================================================
# HISTORY SETTINGS
# ==========================================================

HISTORY_MAX_ITEMS = 5

# ==========================================================
# OUTPUT SETTINGS
# ==========================================================

SCREENSHOT_FOLDER = "output/screenshots"