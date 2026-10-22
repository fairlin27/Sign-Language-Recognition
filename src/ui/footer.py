"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Footer Panel
==========================================================
"""

import customtkinter as ctk
from datetime import datetime

from config import *


class FooterPanel(ctk.CTkFrame):

    # ======================================================
    # Initialize
    # ======================================================

    def __init__(self, parent):

        super().__init__(
            parent,
            height=FOOTER_HEIGHT,
            fg_color=HEADER,
            corner_radius=0
        )

        self.grid_propagate(False)

        # ==================================================
        # Columns
        # ==================================================

        for column in range(4):

            self.grid_columnconfigure(
                column,
                weight=1
            )

        # ==================================================
        # Rows
        # ==================================================

        self.grid_rowconfigure(
            0,
            weight=0
        )

        self.grid_rowconfigure(
            1,
            weight=0
        )

        self.grid_rowconfigure(
            2,
            weight=1
        )

        # ==================================================
        # College
        # ==================================================

        self.college = ctk.CTkLabel(
            self,
            text=COLLEGE_NAME,
            font=("Segoe UI", 14, "bold"),
            text_color=TEXT
        )

        self.college.grid(
            row=0,
            column=0,
            columnspan=4,
            pady=(2, 0)
        )

        # ==================================================
        # Department + Course
        # ==================================================

        self.department = ctk.CTkLabel(
            self,
            text=f"{DEPARTMENT} | {COURSE}",
            font=("Segoe UI", 12),
            text_color=SUBTEXT
        )

        self.department.grid(
            row=1,
            column=0,
            columnspan=4,
            pady=(0, 0)
        )

        # ==================================================
        # Guide
        # ==================================================

        self.guide = ctk.CTkLabel(
            self,
            text=f"Guide: {GUIDE_NAME}",
            font=("Segoe UI", 12),
            text_color=TEXT
        )

        self.guide.grid(
            row=2,
            column=0,
            padx=8,
            pady=(0, 2),
            sticky="w"
        )

        # ==================================================
        # Developer
        # ==================================================

        self.developer = ctk.CTkLabel(
            self,
            text=f"Developed By: {DEVELOPER}",
            font=("Segoe UI", 12),
            text_color=TEXT
        )

        self.developer.grid(
            row=2,
            column=1,
            padx=8,
            pady=(0, 2)
        )

        # ==================================================
        # Version
        # ==================================================

        self.version = ctk.CTkLabel(
            self,
            text=VERSION,
            font=("Segoe UI", 12),
            text_color=TEXT
        )

        self.version.grid(
            row=2,
            column=2,
            padx=8,
            pady=(0, 2)
        )

        # ==================================================
        # Date & Time
        # ==================================================

        self.datetime = ctk.CTkLabel(
            self,
            text="",
            font=("Segoe UI", 12),
            text_color=SUCCESS
        )

        self.datetime.grid(
            row=2,
            column=3,
            padx=8,
            pady=(0, 2),
            sticky="e"
        )

        # ==================================================
        # Start Clock
        # ==================================================

        self.update_time()

    # ======================================================
    # Live Date & Time
    # ======================================================

    def update_time(self):

        current = datetime.now().strftime(
            "%d-%m-%Y   %I:%M:%S %p"
        )

        self.datetime.configure(
            text=current
        )

        self.after(
            1000,
            self.update_time
        )