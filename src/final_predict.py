"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Final Application
==========================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import time
import cv2

from PIL import Image

import customtkinter as ctk

from config import *

from camera import CameraManager
from predictor import Predictor
from speech import SpeechManager

from ui.main_window import MainWindow
from ui.header import HeaderPanel
from ui.camera_panel import CameraPanel
from ui.info_panel import InfoPanel
from ui.history_panel import HistoryPanel
from ui.control_panel import ControlPanel
from ui.footer import FooterPanel


# ==========================================================
# APPLICATION
# ==========================================================

class SignLanguageApp(MainWindow):

    def __init__(self):

        super().__init__()

        print("=" * 60)
        print(PROJECT_NAME)
        print("=" * 60)

        # ------------------------------------------
        # Modules
        # ------------------------------------------

        self.camera = None

        self.predictor = Predictor()

        self.speech = SpeechManager()

        # ------------------------------------------
        # Variables
        # ------------------------------------------

        self.running = False

        self.current_frame = None

        self.last_gesture = ""

        self.fps = 0

        self.prev_time = time.time()

        # ------------------------------------------
        # UI
        # ------------------------------------------

        self.create_ui()

        self.bind_buttons()

        print("✓ Application Ready")

    # ======================================================
    # Create UI
    # ======================================================

    def create_ui(self):

        # Header
        self.header = HeaderPanel(self)

        self.header.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="nsew"
        )

        # Camera
        self.camera_panel = CameraPanel(self)

        self.camera_panel.grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        # Right Panel
        self.right_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.right_frame.grid(
            row=1,
            column=1,
            padx=(0,10),
            pady=10,
            sticky="nsew"
        )

        # Info
        self.info_panel = InfoPanel(
            self.right_frame
        )

        self.info_panel.pack(
            fill="x",
            pady=(0,10)
        )

        # History
        self.history_panel = HistoryPanel(
            self.right_frame
        )

        self.history_panel.pack(
            fill="both",
            expand=True
        )

        # Controls
        self.control_panel = ControlPanel(self)

        self.control_panel.grid(
            row=2,
            column=0,
            columnspan=2,
            padx=10,
            pady=10,
            sticky="ew"
        )

        # Footer
        self.footer = FooterPanel(self)

        self.footer.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew"
        )

    # ======================================================
    # Button Events
    # ======================================================

    def bind_buttons(self):

        self.control_panel.set_start_command(
            self.start_camera
        )

        self.control_panel.set_stop_command(
            self.stop_camera
        )

        self.control_panel.set_clear_command(
            self.clear_history
        )

        self.control_panel.set_save_command(
            self.save_screenshot
        )

        self.control_panel.set_exit_command(
            self.exit_application
        )
        