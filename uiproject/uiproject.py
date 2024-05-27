"C:\\Users\\klinc\\Desktop\\uiproject\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat"
import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
import time
from PIL import Image, ImageTk
import tkinter as tk
import os

# Initialize the mixer for playing alarm sound
print("Initializing mixer for alarm sound...")
mixer.init()
alarm_path = r"C:\Users\klinc\Desktop\uiproject\alarm2.mp3"
mixer.music.load(alarm_path)  # Set the alarm sound file path
print(f"Alarm sound loaded from {alarm_path}")


# Function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    print("Calculating eye aspect ratio (EAR)...")
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    print(f"EAR calculated: {ear}")
    return ear


# Threshold values
EYE_AR_THRESH = 0.23  # Threshold value for detecting closed eyes
EYE_AR_CONSEC_FRAMES = 20  # Minimum number of frames with eyes closed to trigger the alert

# Initialize the frame counter and the total number of blinks
COUNTER = 0
TOTAL_BLINKS = 0  # Total number of blinks (not used, but let's keep it for tracking)

# Get the indexes of the facial landmarks for the left and right eye
print("Getting facial landmark indexes for eyes...")
(ls, le) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rs, re) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
print(f"Left eye landmarks: {ls} to {le}, Right eye landmarks: {rs} to {re}")

# Initialize Dlib's face detector and create the facial landmark predictor
print("Initializing Dlib's face detector and shape predictor...")
detect = dlib.get_frontal_face_detector()
shape_predictor_path = "C:\\Users\\klinc\\Desktop\\uiproject\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat"

# Check if the shape predictor file exists
if not os.path.exists(shape_predictor_path):
    raise FileNotFoundError(
        f"The shape predictor file {shape_predictor_path} was not found. Please ensure the file is available.")

predictor = dlib.shape_predictor(shape_predictor_path)
print(f"Shape predictor loaded from {shape_predictor_path}")

# Initialize the video stream and allow the camera sensor to warm up
print("Starting video stream and allowing camera to warm up...")
cap = cv2.VideoCapture(0)
time.sleep(1.0)

# Initialize Tkinter root and label
print("Setting up Tkinter window...")
root = tk.Tk()
root.title("Eye Blink Detection")
label = tk.Label(root)
label.pack()


def update_frame():
    global COUNTER, TOTAL_BLINKS

    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            return

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)
            lefteye = shape[ls:le]
            righteye = shape[rs:re]
            leftear = eye_aspect_ratio(lefteye)
            rightear = eye_aspect_ratio(righteye)
            ear = (leftear + rightear) / 2.0  # Compute average EAR

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                eye_color = (0, 0, 255)  # Red for closed eyes
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "ALERT!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()  # Play the alarm
                        print("Playing alarm sound!")
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                    print(f"Total blinks: {TOTAL_BLINKS}")
                COUNTER = 0
                eye_color = (0, 255, 0)  # Green for open eyes
                mixer.music.stop()  # Stop the alarm if eyes are open

            leftEyeHull = cv2.convexHull(lefteye)
            rightEyeHull = cv2.convexHull(righteye)
            cv2.drawContours(frame, [leftEyeHull], -1, eye_color, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, eye_color, 1)

            # Debugging: Print EAR and counter values
            print(f"EAR: {ear:.2f}, COUNTER: {COUNTER}")

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert the frame to ImageTk format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=image)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        # Repeat the process after a short delay
        root.after(10, update_frame)

    except Exception as e:
        print(f"Error: {e}")


# Start the frame update
update_frame()
root.mainloop()

# Clean up
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Program ended.")
