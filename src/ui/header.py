"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Header Panel
==========================================================
"""

import customtkinter as ctk

from config import *


class HeaderPanel(ctk.CTkFrame):

    # ======================================================
    # Initialize Header
    # ======================================================

    def __init__(self, parent):

        super().__init__(
            parent,
            height=HEADER_HEIGHT,
            fg_color=HEADER,
            corner_radius=0
        )

        self.grid_propagate(False)

        # ==================================================
        # Header Grid
        # ==================================================

        self.grid_columnconfigure(
            0,
            weight=0
        )

        self.grid_columnconfigure(
            1,
            weight=1
        )

        self.grid_columnconfigure(
            2,
            weight=0
        )

        self.grid_rowconfigure(
            0,
            weight=1
        )

        # ==================================================
        # Logo
        # ==================================================

        self.logo = ctk.CTkLabel(
            self,
            text="🤟",
            font=("Segoe UI Emoji", 38),
            text_color=TEXT
        )

        self.logo.grid(
            row=0,
            column=0,
            padx=(25, 20),
            sticky="w"
        )

        # ==================================================
        # Center Content
        # ==================================================

        self.title_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.title_frame.grid(
            row=0,
            column=1,
            sticky="w"
        )

        # ==================================================
        # Project Title
        # ==================================================

        self.title = ctk.CTkLabel(
            self.title_frame,
            text=PROJECT_NAME,
            font=TITLE_FONT,
            text_color=TEXT
        )

        self.title.pack(
            anchor="w"
        )

        # ==================================================
        # Subtitle
        # ==================================================

        self.subtitle = ctk.CTkLabel(
            self.title_frame,
            text=PROJECT_TITLE,
            font=SUBTITLE_FONT,
            text_color=SUBTEXT
        )

        self.subtitle.pack(
            anchor="w",
            pady=(2, 0)
        )

        # ==================================================
        # Live Status
        # ==================================================

        self.status_frame = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.status_frame.grid(
            row=0,
            column=2,
            padx=(20, 25),
            sticky="e"
        )

        # Status Indicator

        self.status_dot = ctk.CTkLabel(
            self.status_frame,
            text="●",
            font=("Segoe UI", 18),
            text_color=SUCCESS
        )

        self.status_dot.pack(
            side="left",
            padx=(0, 5)
        )

        # Status Text

        self.status = ctk.CTkLabel(
            self.status_frame,
            text="LIVE",
            font=BUTTON_FONT,
            text_color=SUCCESS
        )

        self.status.pack(
            side="left"
        )

    # ======================================================
    # Update Header Status
    # ======================================================

    def set_status(
        self,
        status,
        color=SUCCESS
    ):

        self.status.configure(
            text=str(status),
            text_color=color
        )

        self.status_dot.configure(
            text_color=color
        )

    # ======================================================
    # Set Live Status
    # ======================================================

    def set_live(self):

        self.set_status(
            "LIVE",
            SUCCESS
        )

    # ======================================================
    # Set Stopped Status
    # ======================================================

    def set_stopped(self):

        self.set_status(
            "STOPPED",
            DANGER
        )

    # ======================================================
    # Set Waiting Status
    # ======================================================

    def set_waiting(self):

        self.set_status(
            "WAITING",
            WARNING
        )