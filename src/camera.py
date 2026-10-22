"""
==========================================================
SIGN LANGUAGE RECOGNITION SYSTEM
Camera Manager
==========================================================
"""

import time

import cv2
import mediapipe as mp

from config import *


class CameraManager:

    # ======================================================
    # Initialize Camera
    # ======================================================

    def __init__(self):

        print("Loading Camera...")

        # --------------------------------------------------
        # Open Webcam
        # --------------------------------------------------

        self.cap = cv2.VideoCapture(
            CAMERA_ID
        )

        if not self.cap.isOpened():

            raise Exception(
                "Cannot Open Webcam"
            )

        # --------------------------------------------------
        # Camera Resolution
        # --------------------------------------------------

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        # --------------------------------------------------
        # Camera FPS
        # --------------------------------------------------

        self.cap.set(
            cv2.CAP_PROP_FPS,
            FPS
        )

        # --------------------------------------------------
        # Reduce Camera Buffer
        # --------------------------------------------------

        try:

            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )

        except Exception:

            pass

        # ==================================================
        # MediaPipe Classes
        # ==================================================

        BaseOptions = (
            mp.tasks.BaseOptions
        )

        HandLandmarker = (
            mp.tasks.vision.HandLandmarker
        )

        HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )

        VisionRunningMode = (
            mp.tasks.vision.RunningMode
        )

        # ==================================================
        # MediaPipe Options
        # ==================================================

        options = HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=MODEL_PATH
            ),

            running_mode=(
                VisionRunningMode.VIDEO
            ),

            # Maximum number of hands
            num_hands=NUM_HANDS,

            # Hand detection
            min_hand_detection_confidence=(
                MIN_HAND_DETECTION_CONFIDENCE
            ),

            # Hand presence
            min_hand_presence_confidence=(
                MIN_HAND_PRESENCE_CONFIDENCE
            ),

            # Hand tracking
            min_tracking_confidence=(
                MIN_TRACKING_CONFIDENCE
            )
        )

        # ==================================================
        # Create Hand Landmarker
        # ==================================================

        self.landmarker = (
            HandLandmarker.create_from_options(
                options
            )
        )

        # ==================================================
        # Timestamp
        # ==================================================

        self.last_timestamp = 0

        # ==================================================
        # Previous Hand Count
        # Used for Diagnostic Output
        # ==================================================

        self.last_hand_count = -1

        # ==================================================
        # Hand Connections
        # ==================================================

        self.hand_connections = [

            # Thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            # Index Finger
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            # Middle Finger
            (5, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            # Ring Finger
            (9, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            # Little Finger
            (13, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            # Palm
            (0, 17)
        ]

        print("✓ Camera Ready")

    # ======================================================
    # Read Camera Frame
    # ======================================================

    def read(self):

        if not self.is_open():

            return False, None

        success, frame = (
            self.cap.read()
        )

        if not success:

            return False, None

        # --------------------------------------------------
        # Mirror Camera
        # --------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # IMPORTANT:
        # ui_test.py expects:
        #
        # success, frame = self.camera.read()

        return True, frame

    # ======================================================
    # Detect Hands
    # ======================================================

    def detect(self, frame):

        if frame is None:

            return None

        # --------------------------------------------------
        # Convert BGR → RGB
        # --------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------------
        # Create MediaPipe Image
        # --------------------------------------------------

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )

        # ==================================================
        # Generate Increasing Timestamp
        # ==================================================

        current_timestamp = int(
            time.monotonic() * 1000
        )

        if (
            current_timestamp
            <= self.last_timestamp
        ):

            current_timestamp = (
                self.last_timestamp + 1
            )

        self.last_timestamp = (
            current_timestamp
        )

        # ==================================================
        # MediaPipe Hand Detection
        # ==================================================

        result = (
            self.landmarker.detect_for_video(
                mp_image,
                current_timestamp
            )
        )

        # ==================================================
        # Count Detected Hands
        # ==================================================

        hand_count = 0

        if result is not None:

            if hasattr(
                result,
                "hand_landmarks"
            ):

                hand_count = len(
                    result.hand_landmarks
                )

        # ==================================================
        # Diagnostic Output
        # Print ONLY when count changes
        # ==================================================

        if hand_count != self.last_hand_count:

            print(
                "Detected Hands:",
                hand_count
            )

            self.last_hand_count = (
                hand_count
            )

        return result

    # ======================================================
    # Validate Landmark
    # ======================================================

    def _valid_landmark(
        self,
        landmark
    ):

        if landmark is None:

            return False

        try:

            x = float(
                landmark.x
            )

            y = float(
                landmark.y
            )

            z = float(
                landmark.z
            )

        except (
            TypeError,
            ValueError
        ):

            return False

        # --------------------------------------------------
        # Coordinate Validation
        # --------------------------------------------------

        if x < -0.1 or x > 1.1:

            return False

        if y < -0.1 or y > 1.1:

            return False

        # --------------------------------------------------
        # Invalid Number Check
        # --------------------------------------------------

        if not (
            abs(x) < 10
            and
            abs(y) < 10
            and
            abs(z) < 10
        ):

            return False

        return True

    # ======================================================
    # Draw Hand Landmarks
    # ======================================================

    def draw(
        self,
        frame,
        result
    ):

        if frame is None:

            return frame

        if result is None:

            return frame

        if not hasattr(
            result,
            "hand_landmarks"
        ):

            return frame

        if len(
            result.hand_landmarks
        ) == 0:

            return frame

        # --------------------------------------------------
        # Frame Dimensions
        # --------------------------------------------------

        height, width = (
            frame.shape[:2]
        )

        # ==================================================
        # Draw Every Detected Hand
        # ==================================================

        for hand_landmarks in (
            result.hand_landmarks
        ):

            # --------------------------------------------------
            # A Complete Hand Has 21 Landmarks
            # --------------------------------------------------

            if len(
                hand_landmarks
            ) != 21:

                continue

            # ==================================================
            # Draw Connections
            # ==================================================

            for start, end in (
                self.hand_connections
            ):

                start_landmark = (
                    hand_landmarks[start]
                )

                end_landmark = (
                    hand_landmarks[end]
                )

                if not self._valid_landmark(
                    start_landmark
                ):

                    continue

                if not self._valid_landmark(
                    end_landmark
                ):

                    continue

                # --------------------------------------------------
                # Convert Normalized Coordinates
                # --------------------------------------------------

                x1 = int(
                    start_landmark.x * width
                )

                y1 = int(
                    start_landmark.y * height
                )

                x2 = int(
                    end_landmark.x * width
                )

                y2 = int(
                    end_landmark.y * height
                )

                # --------------------------------------------------
                # Boundary Check
                # --------------------------------------------------

                if not (
                    0 <= x1 < width
                    and
                    0 <= y1 < height
                    and
                    0 <= x2 < width
                    and
                    0 <= y2 < height
                ):

                    continue

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    CONNECTION_COLOR,
                    2,
                    cv2.LINE_AA
                )

            # ==================================================
            # Draw 21 Landmark Points
            # ==================================================

            for landmark in (
                hand_landmarks
            ):

                if not self._valid_landmark(
                    landmark
                ):

                    continue

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                # --------------------------------------------------
                # Boundary Check
                # --------------------------------------------------

                if not (
                    0 <= x < width
                    and
                    0 <= y < height
                ):

                    continue

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    LANDMARK_COLOR,
                    -1,
                    cv2.LINE_AA
                )

        return frame

    # ======================================================
    # Get Hand Count
    # ======================================================

    def get_hand_count(
        self,
        result
    ):

        if result is None:

            return 0

        if not hasattr(
            result,
            "hand_landmarks"
        ):

            return 0

        return len(
            result.hand_landmarks
        )

    # ======================================================
    # Get Landmark Count
    # ======================================================

    def get_landmark_count(
        self,
        result
    ):

        if result is None:

            return 0

        if not hasattr(
            result,
            "hand_landmarks"
        ):

            return 0

        total_landmarks = 0

        for hand in (
            result.hand_landmarks
        ):

            if len(hand) == 21:

                total_landmarks += 21

        return total_landmarks

    # ======================================================
    # Get Last Hand Count
    # ======================================================

    def get_last_hand_count(self):

        return self.last_hand_count

    # ======================================================
    # Camera Status
    # ======================================================

    def is_open(self):

        if self.cap is None:

            return False

        return self.cap.isOpened()

    # ======================================================
    # Close Camera
    # ======================================================

    def close(self):

        # --------------------------------------------------
        # Release Webcam
        # --------------------------------------------------

        if self.cap is not None:

            if self.cap.isOpened():

                self.cap.release()

        # --------------------------------------------------
        # Close MediaPipe
        # --------------------------------------------------

        if self.landmarker is not None:

            try:

                self.landmarker.close()

            except Exception as error:

                print(
                    "Landmarker Close Error:",
                    error
                )

        # --------------------------------------------------
        # OpenCV Cleanup
        # --------------------------------------------------

        cv2.destroyAllWindows()

        # --------------------------------------------------
        # Reset Diagnostic State
        # --------------------------------------------------

        self.last_hand_count = -1

        print("✓ Camera Closed")