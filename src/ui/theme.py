"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Theme Configuration
==========================================================
"""

import customtkinter as ctk

# ==========================================================
# APPEARANCE
# ==========================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==========================================================
# WINDOW
# ==========================================================

WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 850

# ==========================================================
# COLOR PALETTE
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
# BUTTON COLORS
# ==========================================================

BUTTON_COLOR = PRIMARY

BUTTON_HOVER_COLOR = "#1D4ED8"

BUTTON_TEXT_COLOR = TEXT

# ==========================================================
# FONT SETTINGS
# ==========================================================

TITLE_FONT = (
    "Segoe UI",
    28,
    "bold"
)

SUBTITLE_FONT = (
    "Segoe UI",
    15
)

HEADING_FONT = (
    "Segoe UI",
    18,
    "bold"
)

NORMAL_FONT = (
    "Segoe UI",
    15
)

SMALL_FONT = (
    "Segoe UI",
    12
)

BUTTON_FONT = (
    "Segoe UI",
    15,
    "bold"
)

# ==========================================================
# PANEL SETTINGS
# ==========================================================

CORNER_RADIUS = 15

PADDING = 15

CARD_HEIGHT = 65

# ==========================================================
# CAMERA SETTINGS
# ==========================================================

CAMERA_BACKGROUND = "#000000"

CAMERA_BORDER_WIDTH = 2

# ==========================================================
# HISTORY PANEL
# ==========================================================

HISTORY_MAX_ITEMS = 5

# ==========================================================
# STATUS COLORS
# ==========================================================

STATUS_READY = SUCCESS

STATUS_WAITING = WARNING

STATUS_ERROR = DANGER

# ==========================================================
# PROGRESS BAR
# ==========================================================

PROGRESS_COLOR = SUCCESS

PROGRESS_BACKGROUND = SECONDARY