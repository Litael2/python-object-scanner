import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from deepface import DeepFace
import numpy as np

torch = None
try:
    import torch
    import torchvision
    from torchvision import transforms
    from ultralytics import YOLO
except ImportError:
    print("Fehlende Module: torch, torchvision, ultralytics. Installiere mit: pip install torch torchvision ultralytics")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face = mp.solutions.face_detection

def detect_hand_side(landmarks):
    return "Rechts" if landmarks[mp_hands.HandLandmark.WRIST].x < landmarks[mp_hands.HandLandmark.THUMB_CMC].x else "Links"

def estimate_age(frame):
    try:
        result = DeepFace.analyze(frame, actions=["age"], enforce_detection=False)
        return result[0]["age"] if result else "?"
    except:
        return "?"

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb_frame)
    results_face = face_detector.process(rgb_frame)
    
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            hand_side = detect_hand_side(hand_landmarks.landmark)
            h, w, _ = frame.shape
            x, y = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, hand_side, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_roi = frame[y:y+h, x:x+w] if y+h < ih and x+w < iw else None
            age = estimate_age(face_roi) if face_roi is not None and face_roi.size > 0 else "?"
            cv2.putText(frame, f"Alter: {age}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    if torch and model:
        small_frame = cv2.resize(frame, (320, 240))
        results = model(small_frame)
        scale_x, scale_y = frame.shape[1] / 320, frame.shape[0] / 240
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                label = model.names[int(box.cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    lbl_video.imgtk = imgtk
    lbl_video.configure(image=imgtk)
    lbl_video.after(10, process_frame)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

model = YOLO("yolov8n.pt") if torch else None

root = tk.Tk()
root.title("Hand-, Alter- und Objekterkennung")
lbl_video = Label(root)
lbl_video.pack()
process_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
