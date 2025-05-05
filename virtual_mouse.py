import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import tensorflow as tf
import time
from tensorflow.keras.models import load_model

IMAGE_SIZE = 128
MODEL_PATH = 'fingers_detection.keras'

model = load_model(MODEL_PATH)


def preprocess_frame(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    gray = gray.astype(np.float32) / 255.0
    gray = np.reshape(gray, (1, IMAGE_SIZE, IMAGE_SIZE, 1))
    return gray

# Initialize
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
last_click_time = 0

print("Virtual mouse started... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            xmin = max(int(min(x_vals) * w) - 20, 0)
            ymin = max(int(min(y_vals) * h) - 20, 0)
            xmax = min(int(max(x_vals) * w) + 20, w)
            ymax = min(int(max(y_vals) * h) + 20, h)

            center_x = int((xmin + xmax) / 2 / w * screen_w)
            center_y = int((ymin + ymax) / 2 / h * screen_h)
            pyautogui.moveTo(center_x, center_y, duration=0.01)

            ROI_SIZE = 170
            center_x_pixel = int((xmin + xmax) / 2)
            center_y_pixel = int((ymin + ymax) / 2)
            half_roi = ROI_SIZE // 2

            # Ensure ROI stays within frame bounds
            start_x = max(center_x_pixel - half_roi, 0)
            end_x = min(center_x_pixel + half_roi, w)
            start_y = max(center_y_pixel - half_roi, 0)
            end_y = min(center_y_pixel + half_roi, h)

            hand_crop = frame[start_y:end_y, start_x:end_x]
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 255), 2)

            processed = preprocess_frame(hand_crop)
            prediction = model.predict(processed, verbose=0)
            label = "Finger 1" if prediction[0][0] >= 0.5 else "Other"

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            if prediction[0][0] >= 0.5:
                current_time = time.time()
                if current_time - last_click_time > 1:
                    pyautogui.click()
                    print("Click")
                    last_click_time = current_time
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                print("No Action")

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
