"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Control Panel
==========================================================
"""

import customtkinter as ctk

from config import *


class ControlPanel(ctk.CTkFrame):

    # ======================================================
    # Initialize
    # ======================================================

    def __init__(self, parent):

        super().__init__(
            parent,
            height=CONTROL_PANEL_HEIGHT,
            fg_color=CARD,
            corner_radius=15
        )

        self.grid_propagate(False)

        # ==================================================
        # Grid Configuration
        # ==================================================

        for column in range(5):

            self.grid_columnconfigure(
                column,
                weight=1,
                uniform="control"
            )

        self.grid_rowconfigure(
            0,
            weight=1
        )

        # ==================================================
        # Common Button Settings
        # ==================================================

        button_height = 45

        button_width = 170

        # ==================================================
        # Start Button
        # ==================================================

        self.start_btn = ctk.CTkButton(
            self,
            text="▶  Start Camera",
            font=BUTTON_FONT,
            width=button_width,
            height=button_height,
            fg_color=SUCCESS,
            hover_color="#16A34A",
            corner_radius=10
        )

        self.start_btn.grid(
            row=0,
            column=0,
            padx=8,
            pady=15,
            sticky="ew"
        )

        # ==================================================
        # Stop Button
        # ==================================================

        self.stop_btn = ctk.CTkButton(
            self,
            text="⏹  Stop Camera",
            font=BUTTON_FONT,
            width=button_width,
            height=button_height,
            fg_color=WARNING,
            hover_color="#D97706",
            corner_radius=10
        )

        self.stop_btn.grid(
            row=0,
            column=1,
            padx=8,
            pady=15,
            sticky="ew"
        )

        # ==================================================
        # Clear History Button
        # ==================================================

        self.clear_btn = ctk.CTkButton(
            self,
            text="🗑  Clear History",
            font=BUTTON_FONT,
            width=button_width,
            height=button_height,
            fg_color=PRIMARY,
            hover_color="#1D4ED8",
            corner_radius=10
        )

        self.clear_btn.grid(
            row=0,
            column=2,
            padx=8,
            pady=15,
            sticky="ew"
        )

        # ==================================================
        # Save Screenshot Button
        # ==================================================

        self.save_btn = ctk.CTkButton(
            self,
            text="💾  Save Screenshot",
            font=BUTTON_FONT,
            width=button_width,
            height=button_height,
            fg_color=PRIMARY,
            hover_color="#1D4ED8",
            corner_radius=10
        )

        self.save_btn.grid(
            row=0,
            column=3,
            padx=8,
            pady=15,
            sticky="ew"
        )

        # ==================================================
        # Exit Button
        # ==================================================

        self.exit_btn = ctk.CTkButton(
            self,
            text="❌  Exit",
            font=BUTTON_FONT,
            width=button_width,
            height=button_height,
            fg_color=DANGER,
            hover_color="#DC2626",
            corner_radius=10
        )

        self.exit_btn.grid(
            row=0,
            column=4,
            padx=8,
            pady=15,
            sticky="ew"
        )

    # ======================================================
    # Button Callbacks
    # ======================================================

    def set_start_command(self, command):

        self.start_btn.configure(
            command=command
        )

    def set_stop_command(self, command):

        self.stop_btn.configure(
            command=command
        )

    def set_clear_command(self, command):

        self.clear_btn.configure(
            command=command
        )

    def set_save_command(self, command):

        self.save_btn.configure(
            command=command
        )

    def set_exit_command(self, command):

        self.exit_btn.configure(
            command=command
        )

    # ======================================================
    # Enable Buttons
    # ======================================================

    def enable_buttons(self):

        self.start_btn.configure(
            state="normal"
        )

        self.stop_btn.configure(
            state="normal"
        )

        self.clear_btn.configure(
            state="normal"
        )

        self.save_btn.configure(
            state="normal"
        )

        self.exit_btn.configure(
            state="normal"
        )

    # ======================================================
    # Disable Camera Buttons
    # ======================================================

    def disable_camera_buttons(self):

        self.start_btn.configure(
            state="disabled"
        )

        self.stop_btn.configure(
            state="disabled"
        )

    # ======================================================
    # Enable Camera Buttons
    # ======================================================

    def enable_camera_buttons(self):

        self.start_btn.configure(
            state="normal"
        )

        self.stop_btn.configure(
            state="normal"
        )