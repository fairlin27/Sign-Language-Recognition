from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision import HandLandmarkerOptions

MODEL_PATH = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2
)

print("Creating HandLandmarker...")

with HandLandmarker.create_from_options(options) as landmarker:
    print("SUCCESS!")