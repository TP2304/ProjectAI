import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import time
from PIL import Image, ImageTk
import tkinter as tk
import os

# EAR (Eye Aspect Ratio) is used to figure out if eyes are open or shut. It's calculated using distances between points on the eye.
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize mixer for alarm sound
print("Setting up sound stuff...")
mixer.init()
alarm_path = r"C:\Users\klinc\Desktop\uiproject\uiproject\alarm2.mp3"
print(f"Loading alarm from {alarm_path}...")
mixer.music.load(alarm_path)
print("Alarm ready to go.")

# Thresholds
EYE_RATIO_LIMIT = 0.23
FRAME_COUNT_THRESHOLD = 20

# Initialize counters
counter = 0
total_blinks = 0

# Get facial landmarks for eyes
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize Dlib stuff
detector = dlib.get_frontal_face_detector()
shape_predictor_path = r"C:\Users\klinc\Desktop\uiproject\uiproject\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"

if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(f"Shape predictor file {shape_predictor_path} is missing. Please check.")

predictor = dlib.shape_predictor(shape_predictor_path)

# Start video capture and wait for camera to warm up
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Tkinter setup
root = tk.Tk()
root.title("Eye Blink Detection")
label = tk.Label(root)
label.pack()

def update_frame():
    global counter, total_blinks

    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture image")
        root.after(10, update_frame)
        return

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[left_start:left_end]
        right_eye = shape[right_start:right_end]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_RATIO_LIMIT:
            counter += 1
            eye_color = (0, 0, 255)
            if counter >= FRAME_COUNT_THRESHOLD:
                cv2.putText(frame, "WAKE UP!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    print("Playing alarm sound...")
                    mixer.music.play()
        else:
            if counter >= FRAME_COUNT_THRESHOLD:
                total_blinks += 1
            counter = 0
            eye_color = (0, 255, 0)
            mixer.music.stop()

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, eye_color, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, eye_color, 1)

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=image)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()

# QuickHull Algorithm
def quickhull(points):
    def get_side(p1, p2, p):
        return (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    def distance(p1, p2, p):
        return abs((p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0]))

    def add_hull_points(p1, p2, points, hull_points):
        if not points:
            return
        farthest_point = max(points, key=lambda p: distance(p1, p2, p))
        hull_points.append(farthest_point)
        points.remove(farthest_point)
        left_set = [p for p in points if get_side(p1, farthest_point, p) > 0]
        right_set = [p for p in points if get_side(farthest_point, p2, p) > 0]
        add_hull_points(p1, farthest_point, left_set, hull_points)
        add_hull_points(farthest_point, p2, right_set, hull_points)

    if len(points) < 3:
        return points

    min_x_point = min(points, key=lambda p: p[0])
    max_x_point = max(points, key=lambda p: p[0])
    hull_points = [min_x_point, max_x_point]
    left_set = [p for p in points if get_side(min_x_point, max_x_point, p) > 0]
    right_set = [p for p in points if get_side(min_x_point, max_x_point, p) < 0]
    add_hull_points(min_x_point, max_x_point, left_set, hull_points)
    add_hull_points(max_x_point, min_x_point, right_set, hull_points)
    return hull_points

image_path = r'C:\Users\klinc\Desktop\uvrgProjektna\sliki\image_2.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            selected_indices = [
                10, 338, 297, 332, 284, 251, 389, 454, 356, 454, 323, 361, 288, 397, 365,
                379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109
            ]
            landmarks = []
            for idx in selected_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                landmarks.append((x, y))

            if len(landmarks) > 0:
                hull_points = quickhull(landmarks)
                hull_points = sorted(hull_points, key=lambda p: (np.arctan2(p[1] - np.mean([y for _, y in landmarks]), p[0] - np.mean([x for x, _ in landmarks]))))

                for i in range(len(hull_points)):
                    start_point = hull_points[i]
                    end_point = hull_points[(i + 1) % len(hull_points)]
                    cv2.line(image, start_point, end_point, (255, 0, 0), 1)

                for point in landmarks:
                    cv2.circle(image, point, 2, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
