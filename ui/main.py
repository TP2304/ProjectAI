import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition


class FaceEyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.geometry("1000x600")  # Set a fixed window size

        self.image_path = ""
        self.video_path = ""
        self.cap = None

        # Load Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Paths to known faces
        self.known_faces = {
            "Tatjana": r"C:\Users\PC\Desktop\Projekt_z_sliki\slika.jpg",
            "Ana": r"C:\Users\PC\Desktop\Projekt_z_sliki\slika_an.jpg",
            "Stefanija": r"C:\Users\PC\Desktop\Projekt_z_sliki\slika_s.jpg"
        }

        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # Create and arrange widgets
        self.create_widgets()

    def create_widgets(self):
        # Create main frames
        left_frame = tk.Frame(self.root, width=200, padx=10, pady=10, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(self.root, padx=10, pady=10)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Create and place buttons in the left frame
        self.upload_image_button = tk.Button(left_frame, text="Upload Image", command=self.upload_image, width=15)
        self.upload_image_button.pack(pady=10)

        self.upload_video_button = tk.Button(left_frame, text="Upload Video", command=self.upload_video, width=15)
        self.upload_video_button.pack(pady=10)

        self.detect_face_button = tk.Button(left_frame, text="Detect Face", command=self.detect_face, width=15)
        self.detect_face_button.pack(pady=10)

        self.detect_eyes_button = tk.Button(left_frame, text="Detect Eyes", command=self.detect_eyes, width=15)
        self.detect_eyes_button.pack(pady=10)

        self.detect_sunglasses_button = tk.Button(left_frame, text="Detect with Sunglasses",
                                                  command=self.detect_with_sunglasses, width=15)
        self.detect_sunglasses_button.pack(pady=10)

        # Create and place the image label in the right frame
        self.image_label = tk.Label(right_frame, bg="white", relief=tk.SUNKEN, borderwidth=2)
        self.image_label.pack(expand=True, fill=tk.BOTH)

    def load_known_faces(self):
        for name, path in self.known_faces.items():
            image = face_recognition.load_image_file(path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
            else:
                messagebox.showerror("Error", f"Could not locate a face in the reference image for {name}.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            messagebox.showinfo("Image Upload", "Image uploaded successfully!")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.play_video()

    def play_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (600, 600))
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.root.after(10, self.play_video)
        else:
            self.cap.release()

    def detect_face(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = face_recognition.load_image_file(self.image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        image = cv2.imread(self.image_path)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

        messagebox.showinfo("Detection Complete", "Face detection and recognition complete.")

    def detect_eyes(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))

        eyes_detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]  # Consider the full face region
            roi_color = image[y:y + h, x:x + w]  # Match the color ROI to the gray ROI
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            for (ex, ey, ew, eh) in eyes:
                if ey < h // 2:
                    eyes_detected = True
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

        if eyes_detected:
            messagebox.showinfo("Detection Complete", "Eye detection complete.")
        else:
            messagebox.showinfo("Detection Complete", "Eyes not found.")

    def detect_with_sunglasses(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

        messagebox.showinfo("Detection Complete", "Face detection with sunglasses complete.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetectionApp(root)
    root.mainloop()
