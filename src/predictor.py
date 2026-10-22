"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Prediction Manager
==========================================================
"""

import joblib
import numpy as np

from config import *


class Predictor:

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self):

        print("Loading Prediction Model...")

        # --------------------------------------------------
        # Load trained model
        # --------------------------------------------------

        self.model = joblib.load(
            GESTURE_MODEL_PATH
        )

        # --------------------------------------------------
        # Load scaler
        # --------------------------------------------------

        self.scaler = joblib.load(
            SCALER_PATH
        )

        # --------------------------------------------------
        # Load label encoder
        # --------------------------------------------------

        self.label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        # ==================================================
        # CURRENT PREDICTION
        # ==================================================

        self.current_gesture = ""

        self.current_confidence = 0.0

        # ==================================================
        # CANDIDATE STATE
        # ==================================================

        self.candidate_gesture = ""

        self.candidate_frames = 0

        self.candidate_confidence_sum = 0.0

        self.candidate_best_confidence = 0.0

        # ==================================================
        # CONFIRMED STATE
        # ==================================================

        self.confirmed_gesture = ""

        self.confirmed_confidence = 0.0

        # ==================================================
        # EVENT
        # ==================================================

        self.gesture_event = None

        # ==================================================
        # STABILITY
        # ==================================================

        self.required_stable_frames = max(
            8,
            int(STABLE_FRAMES)
        )

        print(
            "✓ Required Stable Frames:",
            self.required_stable_frames
        )

        print("✓ Prediction Module Ready")

    # ======================================================
    # FEATURE EXTRACTION
    # ======================================================

    def extract_features(self, result):

        # --------------------------------------------------
        # 63 features per hand
        # --------------------------------------------------

        left_features = [0.0] * 63

        right_features = [0.0] * 63

        # --------------------------------------------------
        # Invalid result
        # --------------------------------------------------

        if result is None:

            return np.array(
                left_features + right_features,
                dtype=np.float32
            ).reshape(1, -1)

        # --------------------------------------------------
        # MediaPipe result check
        # --------------------------------------------------

        if not hasattr(
            result,
            "hand_landmarks"
        ):

            return np.array(
                left_features + right_features,
                dtype=np.float32
            ).reshape(1, -1)

        # --------------------------------------------------
        # No hands
        # --------------------------------------------------

        if len(
            result.hand_landmarks
        ) == 0:

            return np.array(
                left_features + right_features,
                dtype=np.float32
            ).reshape(1, -1)

        # ==================================================
        # PROCESS HANDS
        # ==================================================

        for hand_landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness
        ):

            current_hand = []

            # --------------------------------------------------
            # Extract 21 landmarks
            # --------------------------------------------------

            for landmark in hand_landmarks:

                current_hand.extend([
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z)
                ])

            # --------------------------------------------------
            # Validate
            # --------------------------------------------------

            if len(current_hand) != 63:

                continue

            # --------------------------------------------------
            # Get handedness
            # --------------------------------------------------

            try:

                hand_name = (
                    handedness[0].category_name
                )

            except Exception:

                hand_name = ""

            hand_name = str(
                hand_name
            ).lower()

            # --------------------------------------------------
            # Left
            # --------------------------------------------------

            if hand_name == "left":

                left_features = current_hand

            # --------------------------------------------------
            # Right
            # --------------------------------------------------

            elif hand_name == "right":

                right_features = current_hand

        # ==================================================
        # COMBINE 126 FEATURES
        # ==================================================

        features = (
            left_features
            +
            right_features
        )

        return np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, result):

        try:

            # --------------------------------------------------
            # Extract
            # --------------------------------------------------

            features = self.extract_features(
                result
            )

            # --------------------------------------------------
            # Scale
            # --------------------------------------------------

            scaled = self.scaler.transform(
                features
            )

            # --------------------------------------------------
            # Model prediction
            # --------------------------------------------------

            prediction = self.model.predict(
                scaled
            )

            # --------------------------------------------------
            # Decode
            # --------------------------------------------------

            gesture = (
                self.label_encoder
                .inverse_transform(
                    prediction
                )[0]
            )

            gesture = str(
                gesture
            ).strip()

            # ==================================================
            # CONFIDENCE
            # ==================================================

            confidence = 0.0

            if hasattr(
                self.model,
                "predict_proba"
            ):

                probabilities = (
                    self.model.predict_proba(
                        scaled
                    )
                )

                confidence = float(
                    np.max(
                        probabilities
                    ) * 100.0
                )

            # --------------------------------------------------
            # Low confidence
            # --------------------------------------------------

            if confidence < MIN_CONFIDENCE:

                return (
                    "Unknown",
                    confidence
                )

            return (
                gesture,
                confidence
            )

        except Exception as error:

            print(
                "Prediction Error:",
                error
            )

            return (
                "Unknown",
                0.0
            )

    # ======================================================
    # PROCESS GESTURE
    # ======================================================

    def process_gesture(
        self,
        gesture,
        confidence
    ):

        # --------------------------------------------------
        # Every call starts with no new event.
        # --------------------------------------------------

        self.gesture_event = None

        # ==================================================
        # INVALID GESTURE
        # ==================================================

        if (
            gesture is None
            or
            gesture == ""
            or
            gesture == "Unknown"
        ):

            self.reset_candidate()

            return None

        # ==================================================
        # CONFIDENCE CHECK
        # ==================================================

        if confidence < MIN_CONFIDENCE:

            self.reset_candidate()

            return None

        gesture = str(
            gesture
        ).strip()

        # ==================================================
        # CURRENT PREDICTION
        # ==================================================

        self.current_gesture = gesture

        self.current_confidence = float(
            confidence
        )

        # ==================================================
        # STATE 1
        # NO CURRENT CANDIDATE
        # ==================================================

        if self.candidate_gesture == "":

            self.start_candidate(
                gesture,
                confidence
            )

            return None

        # ==================================================
        # STATE 2
        # SAME CANDIDATE
        # ==================================================

        if (
            gesture
            ==
            self.candidate_gesture
        ):

            self.candidate_frames += 1

            self.candidate_confidence_sum += (
                float(confidence)
            )

            if (
                confidence
                >
                self.candidate_best_confidence
            ):

                self.candidate_best_confidence = (
                    float(confidence)
                )

        # ==================================================
        # STATE 3
        # DIFFERENT GESTURE
        # ==================================================

        else:

            # --------------------------------------------------
            # IMPORTANT:
            #
            # Previous candidate is discarded.
            # New gesture starts a completely new
            # confirmation cycle.
            # --------------------------------------------------

            self.start_candidate(
                gesture,
                confidence
            )

            return None

        # ==================================================
        # NOT STABLE YET
        # ==================================================

        if (
            self.candidate_frames
            <
            self.required_stable_frames
        ):

            return None

        # ==================================================
        # CANDIDATE IS NOW STABLE
        # ==================================================

        average_confidence = (
            self.candidate_confidence_sum
            /
            self.candidate_frames
        )

        # ==================================================
        # CONFIRM NEW GESTURE
        # ==================================================

        if (
            self.confirmed_gesture
            ==
            ""
            or
            self.candidate_gesture
            !=
            self.confirmed_gesture
        ):

            self.confirmed_gesture = (
                self.candidate_gesture
            )

            self.confirmed_confidence = max(
                average_confidence,
                self.candidate_best_confidence
            )

            # --------------------------------------------------
            # CREATE ONE NEW EVENT
            # --------------------------------------------------

            self.gesture_event = (
                self.confirmed_gesture
            )

            return self.gesture_event

        # ==================================================
        # SAME CONFIRMED GESTURE
        # ==================================================

        return None

    # ======================================================
    # START NEW CANDIDATE
    # ======================================================

    def start_candidate(
        self,
        gesture,
        confidence
    ):

        self.candidate_gesture = str(
            gesture
        ).strip()

        self.candidate_frames = 1

        self.candidate_confidence_sum = (
            float(confidence)
        )

        self.candidate_best_confidence = (
            float(confidence)
        )

    # ======================================================
    # GET EVENT
    # ======================================================

    def get_gesture_event(self):

        event = self.gesture_event

        # --------------------------------------------------
        # Event is consumed once
        # --------------------------------------------------

        self.gesture_event = None

        return event

    # ======================================================
    # STABLE PREDICTION
    # ======================================================

    def get_stable_prediction(
        self,
        gesture
    ):

        return self.process_gesture(
            gesture,
            self.current_confidence
        )

    # ======================================================
    # GET CURRENT GESTURE
    # ======================================================

    def get_current_gesture(self):

        if (
            self.current_gesture
            ==
            ""
        ):

            return None

        return self.current_gesture

    # ======================================================
    # GET CURRENT CONFIDENCE
    # ======================================================

    def get_current_confidence(self):

        return self.current_confidence

    # ======================================================
    # GET CONFIRMED GESTURE
    # ======================================================

    def get_confirmed_gesture(self):

        if (
            self.confirmed_gesture
            ==
            ""
        ):

            return None

        return self.confirmed_gesture

    # ======================================================
    # GET CONFIRMED CONFIDENCE
    # ======================================================

    def get_confirmed_confidence(self):

        return self.confirmed_confidence

    # ======================================================
    # GET CANDIDATE
    # ======================================================

    def get_candidate_gesture(self):

        if (
            self.candidate_gesture
            ==
            ""
        ):

            return None

        return self.candidate_gesture

    # ======================================================
    # GET CANDIDATE FRAMES
    # ======================================================

    def get_candidate_frames(self):

        return self.candidate_frames

    # ======================================================
    # GET REQUIRED FRAMES
    # ======================================================

    def get_required_stable_frames(self):

        return self.required_stable_frames

    # ======================================================
    # RESET CANDIDATE
    # ======================================================

    def reset_candidate(self):

        self.candidate_gesture = ""

        self.candidate_frames = 0

        self.candidate_confidence_sum = 0.0

        self.candidate_best_confidence = 0.0

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        # --------------------------------------------------
        # Current
        # --------------------------------------------------

        self.current_gesture = ""

        self.current_confidence = 0.0

        # --------------------------------------------------
        # Candidate
        # --------------------------------------------------

        self.reset_candidate()

        # --------------------------------------------------
        # Confirmed
        # --------------------------------------------------

        self.confirmed_gesture = ""

        self.confirmed_confidence = 0.0

        # --------------------------------------------------
        # Event
        # --------------------------------------------------

        self.gesture_event = None

    # ======================================================
    # STATUS
    # ======================================================

    def get_status(self):

        return {

            "current_gesture":
                self.current_gesture,

            "current_confidence":
                self.current_confidence,

            "candidate_gesture":
                self.candidate_gesture,

            "candidate_frames":
                self.candidate_frames,

            "required_stable_frames":
                self.required_stable_frames,

            "candidate_best_confidence":
                self.candidate_best_confidence,

            "confirmed_gesture":
                self.confirmed_gesture,

            "confirmed_confidence":
                self.confirmed_confidence,

            "gesture_event":
                self.gesture_event

        }