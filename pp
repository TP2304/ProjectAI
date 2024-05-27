import cv2

def detect_sunglasses(roi_gray, x, y, frame):
    sunglasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    sunglasses = sunglasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    sunglasses_detected = False

    for (sx, sy, sw, sh) in sunglasses:
        if sw > 0 and sh > 0:  # Ensure detected region has a size
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)
            cv2.putText(frame, "Sunglasses Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            sunglasses_detected = True

    return sunglasses_detected

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Failed to load cascades")

SECONDS_TO_RECORD_AFTER_DETECTION = 5
recording_started = False

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, frame_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    sunglasses_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        if detect_sunglasses(roi_gray, x, y, frame):
            sunglasses_detected = True

        if not sunglasses_detected:
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

    if not recording_started and len(faces) > 0:
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, frame_size)
        recording_started = True
        print("Started to record!")
    elif recording_started and len(faces) == 0:
        out.release()
        recording_started = False

    if recording_started:
        out.write(frame)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
