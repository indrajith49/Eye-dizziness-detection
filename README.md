# Eye-dizziness-detection
Used to detect early signs of dizziness and fatigue among employees in the workplace environment
#-----------------------------CODE-----------------------------

import cv2
import numpy as np
from scipy.spatial import distance as dist
import winsound  # for beep sound (works on Windows)


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    # vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


# Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# EAR threshold and frame count
EAR_THRESHOLD = 0.25
FRAME_LIMIT = 15
frame_count = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw eye bounding box
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Approximate eye landmarks (corners + midpoints)
            p1 = (ex, ey + eh // 2)
            p4 = (ex + ew, ey + eh // 2)
            p2 = (ex + ew // 4, ey)
            p3 = (ex + 3 * ew // 4, ey)
            p5 = (ex + 3 * ew // 4, ey + eh)
            p6 = (ex + ew // 4, ey + eh)

            eye_points = [p1, p2, p3, p4, p5, p6]

            # Draw points for visualization
            for point in eye_points:
                cv2.circle(roi_color, point, 2, (0, 0, 255), -1)

            # Calculate EAR
            ear = eye_aspect_ratio(eye_points)

            if ear < EAR_THRESHOLD:
                frame_count += 1
                if frame_count >= FRAME_LIMIT:
                    cv2.putText(frame, "DIZZY ALERT!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    winsound.Beep(1000, 500)  # beep sound
            else:
                frame_count = 0

    cv2.imshow('Eye Fatigue Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
