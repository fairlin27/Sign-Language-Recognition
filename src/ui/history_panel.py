"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Recent History Panel
==========================================================
"""

import customtkinter as ctk

from config import *


class HistoryPanel(ctk.CTkFrame):

    # ======================================================
    # Initialize
    # ======================================================

    def __init__(self, parent):

        super().__init__(
            parent,
            fg_color=CARD,
            corner_radius=15
        )

        self.grid_propagate(False)

        # ==================================================
        # Main Grid
        # ==================================================

        self.grid_columnconfigure(
            0,
            weight=1
        )

        self.grid_rowconfigure(
            0,
            weight=0
        )

        self.grid_rowconfigure(
            1,
            weight=1
        )

        # ==================================================
        # History Title
        # ==================================================

        self.title = ctk.CTkLabel(
            self,
            text="📜 Recent History",
            font=HEADING_FONT,
            text_color=TEXT
        )

        self.title.grid(
            row=0,
            column=0,
            padx=15,
            pady=(6, 4)
        )

        # ==================================================
        # History Container
        # ==================================================

        self.history_frame = ctk.CTkFrame(
            self,
            fg_color=SECONDARY,
            corner_radius=10
        )

        self.history_frame.grid(
            row=1,
            column=0,
            padx=15,
            pady=(0, 8),
            sticky="nsew"
        )

        self.history_frame.grid_columnconfigure(
            0,
            weight=1
        )

        self.history_frame.grid_rowconfigure(
            0,
            weight=1
        )

        # ==================================================
        # Scrollable History
        # ==================================================

        self.history_scroll = ctk.CTkScrollableFrame(
            self.history_frame,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=SECONDARY,
            scrollbar_button_hover_color=BORDER
        )

        self.history_scroll.grid(
            row=0,
            column=0,
            padx=5,
            pady=4,
            sticky="nsew"
        )

        self.history_scroll.grid_columnconfigure(
            0,
            weight=1
        )

        # ==================================================
        # History Data
        # ==================================================

        self.history_items = []

        # ==================================================
        # Initial Message
        # ==================================================

        self.empty_label = ctk.CTkLabel(
            self.history_scroll,
            text="Waiting for Gesture...",
            font=NORMAL_FONT,
            text_color=TEXT
        )

        self.empty_label.grid(
            row=0,
            column=0,
            padx=10,
            pady=6,
            sticky="w"
        )

    # ======================================================
    # Add History
    # ======================================================

    def add_history(
        self,
        gesture
    ):

        if gesture is None:
            return

        gesture = str(
            gesture
        ).strip()

        if not gesture:
            return

        # --------------------------------------------------
        # Ignore Unknown
        # --------------------------------------------------

        if gesture.lower() == "unknown":
            return

        # --------------------------------------------------
        # Remove Initial Message
        # --------------------------------------------------

        if self.empty_label.winfo_exists():

            self.empty_label.destroy()

        # --------------------------------------------------
        # Add Latest Gesture At Beginning
        # --------------------------------------------------

        self.history_items.insert(
            0,
            gesture
        )

        # --------------------------------------------------
        # Maximum History
        # --------------------------------------------------

        max_items = HISTORY_MAX_ITEMS

        if len(self.history_items) > max_items:

            self.history_items = (
                self.history_items[:max_items]
            )

        # --------------------------------------------------
        # Refresh Display
        # --------------------------------------------------

        self.refresh()

    # ======================================================
    # Alias
    # ======================================================

    def add_item(
        self,
        gesture
    ):

        self.add_history(
            gesture
        )

    # ======================================================
    # Refresh History
    # ======================================================

    def refresh(self):

        # --------------------------------------------------
        # Remove Existing Widgets
        # --------------------------------------------------

        for widget in self.history_scroll.winfo_children():

            widget.destroy()

        # --------------------------------------------------
        # No History
        # --------------------------------------------------

        if not self.history_items:

            self.empty_label = ctk.CTkLabel(
                self.history_scroll,
                text="Waiting for Gesture...",
                font=NORMAL_FONT,
                text_color=TEXT
            )

            self.empty_label.grid(
                row=0,
                column=0,
                padx=10,
                pady=6,
                sticky="w"
            )

            return

        # --------------------------------------------------
        # Display History
        # --------------------------------------------------

        for index, gesture in enumerate(
            self.history_items,
            start=1
        ):

            history_label = ctk.CTkLabel(
                self.history_scroll,
                text=f"{index}.  {gesture}",
                font=NORMAL_FONT,
                text_color=TEXT,
                anchor="w"
            )

            history_label.grid(
                row=index - 1,
                column=0,
                padx=10,
                pady=3,
                sticky="ew"
            )

        # --------------------------------------------------
        # Scroll To Top
        # --------------------------------------------------

        self.after(
            10,
            self.scroll_to_top
        )

    # ======================================================
    # Scroll To Top
    # ======================================================

    def scroll_to_top(self):

        try:

            self.history_scroll._parent_canvas.yview_moveto(
                0
            )

        except Exception:
            pass

    # ======================================================
    # Clear History
    # ======================================================

    def clear(self):

        self.history_items.clear()

        self.refresh()

    # ======================================================
    # Get History
    # ======================================================

    def get_history(self):

        return list(
            self.history_items
        )

    # ======================================================
    # Set History
    # ======================================================

    def set_history(
        self,
        items
    ):

        if items is None:

            self.history_items = []

        else:

            self.history_items = [
                str(item)
                for item in items
                if item is not None
                and str(item).strip()
                and str(item).lower() != "unknown"
            ]

        # --------------------------------------------------
        # Limit Items
        # --------------------------------------------------

        self.history_items = (
            self.history_items[:HISTORY_MAX_ITEMS]
        )

        self.refresh()