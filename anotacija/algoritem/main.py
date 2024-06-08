import cv2
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pygame
import os

class DriverMonitoringSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Monitoring System")
        self.root.configure(background="white")
        self.root.geometry("800x600")

        # Initialize pygame for sound playback
        pygame.mixer.init()

        # Load the warning sound
        self.warning_sound = 'code-red-185448.mp3'

        # Create a frame for buttons on the left side
        self.button_frame = tk.Frame(root, bg="white")
        self.button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.upload_video_btn = tk.Button(self.button_frame, text="Upload Video", command=self.upload_video)
        self.upload_video_btn.pack(pady=10)

        self.upload_image_btn = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_image_btn.pack(pady=10)

        self.open_webcam_btn = tk.Button(self.button_frame, text="Open Webcam", command=self.open_webcam)
        self.open_webcam_btn.pack(pady=10)

        self.detect_sunglasses_btn = tk.Button(self.button_frame, text="Detect Sunglasses", command=self.detect_sunglasses)
        self.detect_sunglasses_btn.pack(pady=10)

        # Create a frame for displaying the image or video in the middle
        self.display_frame = tk.Frame(root, bg="white")
        self.display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.panel = tk.Label(self.display_frame)
        self.panel.pack(pady=10)

        self.status_message = ""
        self.font_scale_warning = 1.0
        self.font_scale_status = 0.8
        self.font_thickness = 2

        self.current_file_path = None
        self.is_video = False

        # Check and load cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise IOError("Failed to load cascades")

    def play_warning_sound(self):
        pygame.mixer.music.load(self.warning_sound)
        pygame.mixer.music.play()

    def open_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        SECONDS_TO_RECORD_AFTER_DETECTION = 5
        SECONDS_EYES_CLOSED = 2
        eyes_closed_start_time = None
        is_recording = False
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            eyes_open = False

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

                if len(eyes) == 0:
                    self.play_warning_sound()
                    cv2.putText(frame, "WARNING: SUNGLASSES DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_warning, (0, 0, 255), self.font_thickness, cv2.LINE_AA)
                    messagebox.showwarning("Warning", "WARNING: SUNGLASSES DETECTED")
                else:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                        eyes_open = True

                if eyes_open:
                    self.status_message = "EYES OPEN"
                    eyes_closed_start_time = None
                else:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    elif time.time() - eyes_closed_start_time >= SECONDS_EYES_CLOSED:
                        self.status_message = "EYES CLOSED"
                        messagebox.showwarning("Warning", "EYES CLOSED")

                cv2.putText(frame, self.status_message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_status, (0, 0, 255), self.font_thickness, cv2.LINE_AA)

            if not is_recording and len(faces) > 0:
                video_writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, frame_size)
                is_recording = True
                print("Started to record!")
            elif is_recording and len(faces) == 0:
                if eyes_closed_start_time and time.time() - eyes_closed_start_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    video_writer.release()
                    is_recording = False
                    print("Stopped recording due to no face detected!")

            if is_recording:
                video_writer.write(frame)

            self.display_frame_func(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if is_recording:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    def upload_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            messagebox.showinfo("Selected File", f"Video selected: {file_path}")
            self.current_file_path = file_path
            self.is_video = True

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            messagebox.showinfo("Selected File", f"Image selected: {file_path}")
            self.current_file_path = file_path
            self.is_video = False

    def process_video(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        SECONDS_EYES_CLOSED = 2
        eyes_closed_start_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            eyes_open = False

            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y + h, x:x + w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

                if len(eyes) == 0:
                    self.play_warning_sound()
                    cv2.putText(frame, "WARNING: SUNGLASSES DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_warning, (0, 0, 255), self.font_thickness, cv2.LINE_AA)
                    messagebox.showwarning("Warning", "WARNING: SUNGLASSES DETECTED")
                else:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                        eyes_open = True

                if eyes_open:
                    self.status_message = "EYES OPEN"
                    eyes_closed_start_time = None
                else:
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    elif time.time() - eyes_closed_start_time >= SECONDS_EYES_CLOSED:
                        self.status_message = "EYES CLOSED"
                        messagebox.showwarning("Warning", "EYES CLOSED")

                cv2.putText(frame, self.status_message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_status, (0, 0, 255), self.font_thickness, cv2.LINE_AA)

            self.display_frame_func(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            raise IOError("Cannot open image file")

        SECONDS_EYES_CLOSED = 2
        eyes_closed_start_time = None

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)

        eyes_open = False

        for (x, y, w, h) in faces:
            roi_gray = gray_image[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            if len(eyes) == 0:
                self.play_warning_sound()
                cv2.putText(image, "WARNING: SUNGLASSES DETECTED", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_warning, (0, 0, 255), self.font_thickness, cv2.LINE_AA)
                messagebox.showwarning("Warning", "WARNING: SUNGLASSES DETECTED")
            else:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
                    eyes_open = True

            if eyes_open:
                self.status_message = "EYES OPEN"
                eyes_closed_start_time = None
            else:
                if eyes_closed_start_time is None:
                    eyes_closed_start_time = time.time()
                elif time.time() - eyes_closed_start_time >= SECONDS_EYES_CLOSED:
                    self.status_message = "EYES CLOSED"
                    messagebox.showwarning("Warning", "EYES CLOSED")

            cv2.putText(image, self.status_message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale_status, (0, 0, 255), self.font_thickness, cv2.LINE_AA)

        self.display_frame_func(image, image=True)

    def detect_sunglasses(self):
        if not hasattr(self, 'current_file_path') or self.current_file_path is None:
            messagebox.showwarning("Warning", "Please upload an image or video first.")
            return

        if self.is_video:
            self.process_video(self.current_file_path)
        else:
            self.process_image(self.current_file_path)

    def display_frame_func(self, frame, image=False):
        frame = cv2.resize(frame, (600, 600))
        if image:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

def create_gui():
    root = tk.Tk()
    app = DriverMonitoringSystem(root)
    root.mainloop()

create_gui()
