"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Detection Information Panel
==========================================================
"""

import customtkinter as ctk

from config import *


class InfoPanel(ctk.CTkFrame):

    # ======================================================
    # Initialize
    # ======================================================

    def __init__(self, parent):

        super().__init__(
            parent,
            width=INFO_PANEL_WIDTH,
            height=INFO_PANEL_HEIGHT,
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

        for row in range(1, 6):

            self.grid_rowconfigure(
                row,
                weight=0
            )

        # ==================================================
        # Detection Information Title
        # ==================================================

        self.title = ctk.CTkLabel(
            self,
            text="▣ Detection Information",
            font=HEADING_FONT,
            text_color=TEXT
        )

        self.title.grid(
            row=0,
            column=0,
            padx=15,
            pady=(8, 5),
            sticky="w"
        )

        # ==================================================
        # 1. Gesture
        # ==================================================

        self.gesture = self.create_card(
            row=1,
            title="♧ Gesture",
            value="Waiting..."
        )

        # ==================================================
        # 2. Confidence
        # ==================================================

        self.confidence = self.create_card(
            row=2,
            title="▧ Confidence",
            value="0.00%"
        )

        # ==================================================
        # 3. Hands Detected
        # ==================================================

        self.hands = self.create_card(
            row=3,
            title="♧ Hands Detected",
            value="0 / 2"
        )

        # ==================================================
        # 4. Landmarks
        # ==================================================

        self.landmarks = self.create_card(
            row=4,
            title="◉ Landmarks",
            value="0 / 42"
        )

        # ==================================================
        # 5. Speech
        # ==================================================

        self.speech = self.create_card(
            row=5,
            title="◖ Speech",
            value="Ready"
        )

    # ======================================================
    # Create Card
    # ======================================================

    def create_card(
        self,
        row,
        title,
        value
    ):

        card = ctk.CTkFrame(
            self,
            height=50,
            fg_color=SECONDARY,
            corner_radius=10
        )

        card.grid(
            row=row,
            column=0,
            padx=15,
            pady=2,
            sticky="ew"
        )

        card.grid_propagate(False)

        # ==================================================
        # Card Grid
        # ==================================================

        card.grid_columnconfigure(
            0,
            weight=1
        )

        card.grid_rowconfigure(
            0,
            weight=0
        )

        card.grid_rowconfigure(
            1,
            weight=1
        )

        # ==================================================
        # Title
        # ==================================================

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=SMALL_FONT,
            text_color=SUBTEXT
        )

        title_label.grid(
            row=0,
            column=0,
            padx=14,
            pady=(3, 0),
            sticky="w"
        )

        # ==================================================
        # Value
        # ==================================================

        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=HEADING_FONT,
            text_color=SUCCESS
        )

        value_label.grid(
            row=1,
            column=0,
            padx=14,
            pady=(0, 2),
            sticky="w"
        )

        return value_label

    # ======================================================
    # Update Information
    # ======================================================

    def update_info(
        self,
        gesture="Waiting...",
        confidence=0,
        hands=0,
        speech="Ready",
        status="Idle",
        fps=0,
        landmarks=None
    ):

        # ==================================================
        # Gesture
        # ==================================================

        self.gesture.configure(
            text=str(gesture)
        )

        # ==================================================
        # Confidence
        # ==================================================

        try:

            confidence_value = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            confidence_value = 0.0

        self.confidence.configure(
            text=f"{confidence_value:.2f}%"
        )

        # ==================================================
        # Confidence Color
        # ==================================================

        if confidence_value >= 85:

            self.confidence.configure(
                text_color=SUCCESS
            )

        elif confidence_value >= 60:

            self.confidence.configure(
                text_color=WARNING
            )

        else:

            self.confidence.configure(
                text_color=SUBTEXT
            )

        # ==================================================
        # Hands
        # ==================================================

        try:

            hand_count = int(
                hands
            )

        except (
            TypeError,
            ValueError
        ):

            hand_count = 0

        hand_count = max(
            0,
            min(
                hand_count,
                2
            )
        )

        self.hands.configure(
            text=f"{hand_count} / 2"
        )

        # ==================================================
        # Landmarks
        # ==================================================

        if landmarks is None:

            landmark_count = (
                hand_count * 21
            )

        else:

            try:

                landmark_count = int(
                    landmarks
                )

            except (
                TypeError,
                ValueError
            ):

                landmark_count = (
                    hand_count * 21
                )

        landmark_count = max(
            0,
            min(
                landmark_count,
                42
            )
        )

        self.landmarks.configure(
            text=f"{landmark_count} / 42"
        )

        # ==================================================
        # Speech
        # ==================================================

        self.speech.configure(
            text=str(speech)
        )

    # ======================================================
    # Set Gesture
    # ======================================================

    def set_gesture(
        self,
        gesture
    ):

        self.gesture.configure(
            text=str(gesture)
        )

    # ======================================================
    # Set Confidence
    # ======================================================

    def set_confidence(
        self,
        confidence
    ):

        try:

            value = float(
                confidence
            )

        except (
            TypeError,
            ValueError
        ):

            value = 0.0

        self.confidence.configure(
            text=f"{value:.2f}%"
        )

    # ======================================================
    # Set Hands
    # ======================================================

    def set_hands(
        self,
        hands
    ):

        try:

            count = int(
                hands
            )

        except (
            TypeError,
            ValueError
        ):

            count = 0

        count = max(
            0,
            min(
                count,
                2
            )
        )

        self.hands.configure(
            text=f"{count} / 2"
        )

        self.landmarks.configure(
            text=f"{count * 21} / 42"
        )

    # ======================================================
    # Set Landmarks
    # ======================================================

    def set_landmarks(
        self,
        landmarks
    ):

        try:

            count = int(
                landmarks
            )

        except (
            TypeError,
            ValueError
        ):

            count = 0

        count = max(
            0,
            min(
                count,
                42
            )
        )

        self.landmarks.configure(
            text=f"{count} / 42"
        )

    # ======================================================
    # Set Speech
    # ======================================================

    def set_speech(
        self,
        speech
    ):

        self.speech.configure(
            text=str(speech)
        )

    # ======================================================
    # Reset Panel
    # ======================================================

    def reset(self):

        self.update_info(
            gesture="Waiting...",
            confidence=0,
            hands=0,
            speech="Ready",
            status="Idle",
            fps=0,
            landmarks=0
        )