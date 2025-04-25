import os
import numpy as np
from pygame import mixer
import time
import cv2
import tensorflow as tf

# Prevent using GPU (if not available or not needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the trained model
model = tf.keras.models.load_model('models/cnncat2.h5', compile=False)
model.save('models/saved_model', save_format='tf')
model = tf.keras.models.load_model('models/saved_model')

# Initialize the alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade files for face and eye detection
face = cv2.CascadeClassifier(r'haar_cascade_files//haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar_cascade_files//haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar_cascade_files//haarcascade_righteye_2splits.xml')

# Check if cascade file exists
if not os.path.exists('haar_cascade_files/haarcascade_frontalface_alt.xml'):
    raise FileNotFoundError("Cascade file for face detection not found!")

# Labels
lbl = ['Close', 'Open']

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam not detected or could not be initialized.")

# Setup
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
alarm_on = False  # To prevent continuous alarm

# Main loop
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        rpred = np.argmax(model.predict(r_eye), axis=1)
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye), axis=1)
        break

    # If both eyes are closed
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Drowsiness alert
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)

        if not alarm_on:
            try:
                sound.play()
                alarm_on = True
            except Exception as e:
                print(f"Error playing alarm sound: {e}")

        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2

        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:
        alarm_on = False  # Reset alarm state if user is alert

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
