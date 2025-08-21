# Eye-dizziness-detection
Used to detect early signs of dizziness and fatigue among employees in the workplace environment
#-----------------------------CODE-----------------------------

import cv2
import winsound
from collections import deque
#Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#parameters
CONSEC_FRAMES = 15  # frames before alert
EAR_THRESHOLD = 0.25  # ratio threshold for “closed” eye
counter = 0

#smooth EAR over last N frames
SMOOTH_FRAMES = 5
ear_history = deque(maxlen=SMOOTH_FRAMES)

#beep parameters
FREQ = 1000
DUR = 500

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    ear_values = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            ear = eh / ew
            ear_values.append(ear)

    #Calculate average EAR for smoothing
    if ear_values:
        avg_ear = sum(ear_values) / len(ear_values)
    else:
        avg_ear = 0  # no eyes detected → treat as closed

    ear_history.append(avg_ear)
    smooth_ear = sum(ear_history) / len(ear_history)

    #triggers the alert
    if smooth_ear < EAR_THRESHOLD:
        counter += 1
        if counter >= CONSEC_FRAMES:
            cv2.putText(frame, "DIZZY ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            winsound.Beep(FREQ, DUR)
    else:
        counter = 0

    cv2.imshow("Eye Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
