# app.py
from flask import Flask, render_template, Response
import threading
from typing import List
import cv2
import numpy as np
import os
from flask import request
from flask import jsonify

app = Flask(__name__)

# Video capture object
camera = None
SEQ_LEN = 30
SEQUENCE = []
LATEST_RESULT = None
THREADS = []

def get_latest_result():
    global LATEST_RESULT
    if LATEST_RESULT is None:
        result = 'Waiting for result ... ...'
    else:
        result = LATEST_RESULT
    return {'result-text': result}

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

def send_video_to_api(video: List):
    global LATEST_RESULT
    def dummy_function(video):
        # Dummy function to simulate API call, add your logic here
        dummy_msg = ['Test', 'Test2', 'Test3']
        import random
        return random.choice(dummy_msg)
    LATEST_RESULT = dummy_function(video)

def process_sequence(frame):
    global SEQUENCE, SEQ_LEN
    seq, lim = SEQUENCE, SEQ_LEN
    if len(seq) < lim:
        seq.append(frame)
    elif len(seq) == lim:
        global THREADS
        cur_thread = threading.Thread(target=send_video_to_api, args=(seq,))
        THREADS.append(cur_thread)
        cur_thread.start()
        seq.pop(0)
        seq.append(frame)
    else:
        raise Exception("Sequence length exceeded")

# Generator function that yields frames
def generate_frames(effect=None):
    camera = get_camera()

    while True:
        success, frame = camera.read()
        if not success:
            error_message = b'--frame\r\n' \
                           b'Content-Type: text/plain\r\n\r\n' \
                           b'error_loading_frame\r\n'
            yield error_message
            continue
        
        # Apply effects based on parameter
        frame = apply_effect(frame, effect)
        model = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        faces = model.detectMultiScale(frame)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Encode the processed frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        process_sequence(frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        
@app.route('/result')
def results():
    return Response(get_latest_result(), 
                   mimetype='application/json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Default effect
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<effect>')
def video_feed_effect(effect):
    return Response(generate_frames(effect),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Clean up resources when the server shuts down
@app.teardown_appcontext
def teardown_camera(exception):
    release_camera()

if __name__ == '__main__':
    app.run(debug=True)