"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Camera Display Panel
==========================================================
"""

import customtkinter as ctk

from PIL import Image, ImageTk

from config import *


class CameraPanel(ctk.CTkFrame):

    # ======================================================
    # Initialize Camera Panel
    # ======================================================

    def __init__(self, parent):

        super().__init__(
            parent,
            width=CAMERA_PANEL_WIDTH,
            height=CAMERA_PANEL_HEIGHT,
            fg_color=CARD,
            corner_radius=15
        )

        # Keep fixed panel size
        self.grid_propagate(False)

        # ==================================================
        # Camera Display Size
        # ==================================================

        self.display_width = 760
        self.display_height = 430

        # ==================================================
        # Title
        # ==================================================

        self.title = ctk.CTkLabel(
            self,
            text="📷 Live Camera",
            font=HEADING_FONT,
            text_color=TEXT
        )

        self.title.pack(
            pady=(15, 8)
        )

        # ==================================================
        # Camera Display Frame
        # ==================================================

        self.display_frame = ctk.CTkFrame(
            self,
            width=self.display_width,
            height=self.display_height,
            fg_color="#000000",
            corner_radius=10
        )

        self.display_frame.pack(
            padx=20,
            pady=8
        )

        self.display_frame.pack_propagate(False)

        # ==================================================
        # Camera Label
        # ==================================================

        self.camera_label = ctk.CTkLabel(
            self.display_frame,
            text="Waiting for Camera...",
            width=self.display_width,
            height=self.display_height,
            fg_color="#000000",
            corner_radius=10,
            text_color=TEXT,
            font=NORMAL_FONT
        )

        self.camera_label.pack(
            fill="both",
            expand=True
        )

        # ==================================================
        # IMPORTANT IMAGE REFERENCES
        # ==================================================

        self.current_image = None
        self.camera_label.image = None

    # ======================================================
    # Update Camera Frame
    # ======================================================

    def update_frame(self, frame):

        if frame is None:
            return

        try:

            # --------------------------------------------------
            # Convert NumPy Array to PIL Image
            # --------------------------------------------------

            image = Image.fromarray(frame)

            # --------------------------------------------------
            # Preserve Original Aspect Ratio
            # --------------------------------------------------

            image.thumbnail(
                (
                    self.display_width,
                    self.display_height
                ),
                Image.Resampling.LANCZOS
            )

            # --------------------------------------------------
            # Create Black Background
            # --------------------------------------------------

            background = Image.new(
                "RGB",
                (
                    self.display_width,
                    self.display_height
                ),
                "black"
            )

            # --------------------------------------------------
            # Center Camera Image
            # --------------------------------------------------

            x_position = (
                self.display_width - image.width
            ) // 2

            y_position = (
                self.display_height - image.height
            ) // 2

            background.paste(
                image,
                (
                    x_position,
                    y_position
                )
            )

            # --------------------------------------------------
            # Create Tkinter Image
            # --------------------------------------------------

            photo = ImageTk.PhotoImage(
                background
            )

            # --------------------------------------------------
            # Keep Strong References
            # --------------------------------------------------

            self.current_image = photo

            self.camera_label.image = photo

            # --------------------------------------------------
            # Update Camera Display
            # --------------------------------------------------

            self.camera_label.configure(
                image=photo,
                text=""
            )

        except Exception as error:

            print(
                "Camera Display Error:",
                error
            )

    # ======================================================
    # Clear Camera Panel
    # ======================================================

    def clear(self):

        try:

            # --------------------------------------------------
            # Remove Current Image
            # --------------------------------------------------

            self.camera_label.configure(
                image="",
                text="Waiting for Camera..."
            )

            # --------------------------------------------------
            # Remove Image References
            # --------------------------------------------------

            self.current_image = None
            self.camera_label.image = None

        except Exception as error:

            print(
                "Camera Clear Error:",
                error
            )