import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import dlib
from pygame import mixer
import os
from imutils import face_utils

class FaceEyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.geometry("1200x600")
        self.cap = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.known_faces = {
            "Tatjana": r"D:\Users\Stefi\Desktop\pp1\pp1\Projekt_z_sliki\slika_t.jpg",
            "Ana": r"D:\Users\Stefi\Desktop\pp1\pp1\Projekt_z_sliki\slika_an.jpg",
            "Stefanija": r"D:\Users\Stefi\Desktop\pp1\pp1\Projekt_z_sliki\slika_s.jpg"
        }
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor_path = r"D:\Users\Stefi\Desktop\pp1\pp1\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.shape_predictor_path):
            raise FileNotFoundError(f"The shape predictor file {self.shape_predictor_path} was not found.")
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)

        mixer.init()
        self.alarm_path = r"D:\Users\Stefi\Desktop\pp1\pp1\alarm2.mp3"
        mixer.music.load(self.alarm_path)

        self.EYE_AR_THRESH = 0.23
        self.EYE_AR_CONSEC_FRAMES = 20
        self.EYE_CLOSE_DURATION = 3

        self.counter = 0
        self.total_blinks = 0
        self.close_start_time = None

        (self.left_start, self.left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.right_start, self.right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        self.nose_tip_idx = (30, 35)
        self.nose_bridge_idx = (27, 30)
        self.HEAD_ORIENTATION_THRESH = 15

        self.create_widgets()

    def create_widgets(self):
        self.font_title = ("Helvetica", 18, "bold")
        self.font_button = ("Helvetica", 12)
        self.font_checkbox = ("Helvetica", 12)
        self.font_text = ("Helvetica", 10)

        self.bg_color = "#d3d3d3"
        self.btn_color = "#a9a9a9"
        self.text_color = "#000000"
        self.frame_color = "#d3d3d3"
        self.canvas_color = "#ffffff"

        self.root.configure(bg=self.bg_color)

        left_frame = tk.Frame(self.root, width=200, padx=10, pady=10, bg=self.frame_color, relief=tk.RIDGE, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(self.root, padx=10, pady=10, bg=self.bg_color)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.upload_image_button = tk.Button(left_frame, text="Upload Image", command=self.upload_image, width=20,
                                             bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.upload_image_button.pack(pady=10)

        self.upload_video_button = tk.Button(left_frame, text="Upload Video", command=self.upload_video, width=20,
                                             bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.upload_video_button.pack(pady=10)

        self.detect_face_button = tk.Button(left_frame, text="Detect Face", command=self.detect_face, width=20,
                                            bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.detect_face_button.pack(pady=10)

        self.detect_eyes_button = tk.Button(left_frame, text="Detect Eyes", command=self.detect_eyes, width=20,
                                            bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.detect_eyes_button.pack(pady=10)

        self.detect_sunglasses_button = tk.Button(left_frame, text="Detect with Sunglasses",
                                                  command=self.detect_with_sunglasses, width=20, bg=self.btn_color,
                                                  fg=self.text_color, font=self.font_button)
        self.detect_sunglasses_button.pack(pady=10)

        self.sunglasses_check_var = tk.IntVar()
        self.eyes_check_var = tk.IntVar()
        self.face_check_var = tk.IntVar()

        self.sunglasses_checkbox = tk.Checkbutton(left_frame, text="Check Sunglasses",
                                                  variable=self.sunglasses_check_var, bg=self.frame_color,
                                                  fg=self.text_color, font=self.font_checkbox)
        self.sunglasses_checkbox.pack(pady=5)

        self.eyes_checkbox = tk.Checkbutton(left_frame, text="Check Eyes", variable=self.eyes_check_var,
                                            bg=self.frame_color, fg=self.text_color, font=self.font_checkbox)
        self.eyes_checkbox.pack(pady=5)

        self.face_checkbox = tk.Checkbutton(left_frame, text="Check Face", variable=self.face_check_var,
                                            bg=self.frame_color, fg=self.text_color, font=self.font_checkbox)
        self.face_checkbox.pack(pady=5)

        self.canvas = tk.Canvas(right_frame, bg=self.canvas_color, relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.message_box = tk.Text(right_frame, width=40, height=10, state=tk.DISABLED, bg=self.canvas_color,
                                   fg=self.text_color, font=self.font_text)
        self.message_box.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    def log_message(self, message):
        self.message_box.configure(state=tk.NORMAL)
        current_text = self.message_box.get("1.0", tk.END)
        if message + "\n" not in current_text:
            self.message_box.insert(tk.END, message + "\n")
            self.message_box.yview(tk.END)
        self.message_box.configure(state=tk.DISABLED)

    def load_known_faces(self):
        for name, path in self.known_faces.items():
            image = face_recognition.load_image_file(path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
            else:
                self.log_message(f"Error: Could not locate a face in the reference image for {name}.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            self.log_message("Image uploaded successfully!")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_delay = int(1000 / fps)
            self.log_message("Video uploaded successfully!")
            self.play_video()

    def play_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (600, 600))
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            self.root.after(self.frame_delay, self.play_video)
        else:
            self.cap.release()

    def detect_face(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.display_and_log(image, "Faces detected successfully.")

    def detect_eyes(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(image, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            self.display_and_log(image, "Eyes detected successfully.")

    def detect_with_sunglasses(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) == 0:
                    cv2.putText(image, "Sunglasses Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            self.display_and_log(image, "Sunglasses detection completed.")

    def display_and_log(self, image, log_message):
        cv2.imshow(log_message, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.log_message(log_message)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetectionApp(root)
    root.mainloop()
