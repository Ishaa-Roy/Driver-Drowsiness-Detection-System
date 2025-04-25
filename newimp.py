import os
import cv2
import time
import numpy as np
import pyttsx3
import dlib
import csv
from imutils import face_utils
from pygame import mixer
from tensorflow.keras.models import load_model
from datetime import datetime

# === SETTINGS ===
ALERT_THRESHOLD = 15
BREAK_REMINDER_INTERVAL = 3600  # seconds (1 hour)

# === INITIALIZE ===
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = load_model('models/cnncat2.h5', compile=False)
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')
engine = pyttsx3.init()

face = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar_cascade_files/haarcascade_righteye_2splits.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('haar_cascade_files/shape_predictor_68_face_landmarks.dat')

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# === SESSION LOGGING ===
def log_event(event):
    with open("sleep_log.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), event])
 
# === START VIDEO ===
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
rpred = [99]
lpred = [99]
start_time = time.time()

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Eye prediction
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye), axis=1)
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Eyes Closed", (10, height - 20), font, 1, (255, 255, 255), 1)
    else:
        score -= 1
        cv2.putText(frame, "Eyes Open", (10, height - 20), font, 1, (255, 255, 255), 1)
    score = max(score, 0)

    # === YAWN DETECTION ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)
        if mar > 0.7:
            cv2.putText(frame, "Yawning!", (width - 150, height - 20), font, 1, (0, 0, 255), 2)
            engine.say("You seem tired. Please take a break.")
            engine.runAndWait()
            log_event("Yawn detected")

    # === ALERT IF TOO SLEEPY ===
    if score > ALERT_THRESHOLD:
        cv2.imwrite(os.path.join(os.getcwd(), 'drowsy.jpg'), frame)
        try:
            alarm_sound.play()
            engine.say("Wake up! You're feeling sleepy.")
            engine.runAndWait()
        except:
            pass
        log_event("Drowsiness detected")
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        thicc = min(16, thicc + 2)
    else:
        thicc = max(2, thicc - 2)

    # === BREAK REMINDER ===
    elapsed = time.time() - start_time
    if elapsed > BREAK_REMINDER_INTERVAL:
        engine.say("You've been working for a while. Time for a 5 minute break!")
        engine.runAndWait()
        log_event("Break reminder")
        start_time = time.time()

    # === DISPLAY ===
    cv2.putText(frame, 'Score:' + str(score), (120, height - 20), font, 1, (255, 255, 255), 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
