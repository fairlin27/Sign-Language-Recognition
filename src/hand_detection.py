import cv2
import mediapipe as mp

# Path to the downloaded MediaPipe model
MODEL_PATH = "models/hand_landmarker.task"

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand Landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,

    # Detect maximum 2 hands
    num_hands=2,

    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hand connections for drawing lines
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Open webcam
cap = cv2.VideoCapture(0)

# Create Hand Landmarker
with HandLandmarker.create_from_options(options) as landmarker:

    frame_timestamp_ms = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera not found")
            break

        # Flip camera like a mirror
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Increase timestamp
        frame_timestamp_ms += 33

        # Detect hand landmarks
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        # Check if hands are detected
        if result.hand_landmarks:

            # Display number of detected hands
            hand_count = len(result.hand_landmarks)

            cv2.putText(
                frame,
                "Hands Detected: " + str(hand_count),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Process each detected hand
            for hand_landmarks in result.hand_landmarks:

                height, width, _ = frame.shape

                # Draw 21 landmark points
                for landmark in hand_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections between landmarks
                for start, end in HAND_CONNECTIONS:

                    x1 = int(
                        hand_landmarks[start].x * width
                    )

                    y1 = int(
                        hand_landmarks[start].y * height
                    )

                    x2 = int(
                        hand_landmarks[end].x * width
                    )

                    y2 = int(
                        hand_landmarks[end].y * height
                    )

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

        else:

            # Display when no hand is detected
            cv2.putText(
                frame,
                "No Hand Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display output
        cv2.imshow(
            "Two Hand Landmark Detection",
            frame
        )

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release webcam
cap.release()
cv2.destroyAllWindows()
