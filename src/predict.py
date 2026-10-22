import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_PATH="models/hand_landmarker.task"
gesture_model=joblib.load("models/gesture_model.pkl")
scaler=joblib.load("models/scaler.pkl")
label_encoder=joblib.load("models/label_encoder.pkl")

BaseOptions=mp.tasks.BaseOptions
HandLandmarker=mp.tasks.vision.HandLandmarker
HandLandmarkerOptions=mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode=mp.tasks.vision.RunningMode

options=HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS=[
(0,1),(1,2),(2,3),(3,4),
(0,5),(5,6),(6,7),(7,8),
(5,9),(9,10),(10,11),(11,12),
(9,13),(13,14),(14,15),(15,16),
(13,17),(17,18),(18,19),(19,20),
(0,17)
]

cap=cv2.VideoCapture(0)
with HandLandmarker.create_from_options(options) as landmarker:
    t=0
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        t+=33
        result=landmarker.detect_for_video(img,t)
        cv2.putText(frame,f"Hands : {len(result.hand_landmarks)}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        if len(result.hand_landmarks)>0:
            h,w,_=frame.shape
            for hand in result.hand_landmarks:
                feat=[]
                for lm in hand:
                    x=int(lm.x*w); y=int(lm.y*h)
                    cv2.circle(frame,(x,y),5,(0,255,0),-1)
                    feat.extend([lm.x,lm.y,lm.z])
                for s,e in HAND_CONNECTIONS:
                    cv2.line(frame,(int(hand[s].x*w),int(hand[s].y*h)),(int(hand[e].x*w),int(hand[e].y*h)),(255,0,0),2)
                while len(feat)<126:
                    feat.append(0.0)
                feat=np.array(feat).reshape(1,-1)
                feat=scaler.transform(feat)
                pred=gesture_model.predict(feat)
                gesture=label_encoder.inverse_transform(pred)[0]
                cv2.putText(frame,"Gesture : "+gesture,(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.imshow("Sign Language Prediction",frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
cap.release()
cv2.destroyAllWindows()