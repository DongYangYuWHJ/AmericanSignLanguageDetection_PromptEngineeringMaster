import cv2
import os
import numpy as np

# Video capture object
camera = None

def get_camera(resolution=(640, 480)):
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # 0 is usually the default webcam
        # Set resolution (optional)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def apply_effect(frame, effect=None):
    if effect == 'grayscale':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
    elif effect == 'blur':
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    elif effect == 'edge':
        frame = cv2.Canny(frame, 100, 200)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
    elif effect == 'invert':
        frame = cv2.bitwise_not(frame)
    elif effect == 'resize_small':
        frame = cv2.resize(frame, (320, 240))
        # Create a black canvas with original dimensions
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Center the resized image
        y_offset = (480 - 240) // 2
        x_offset = (640 - 320) // 2
        canvas[y_offset:y_offset+240, x_offset:x_offset+320] = frame
        frame = canvas
    return frame

def draw_reactangle(frame):
    model = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    faces = model.detectMultiScale(frame)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        