# app.py
from flask import Flask, render_template, Response
import threading

from camera_setting import *
from video_process_utils import *

app = Flask(__name__)

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
        
        process_sequence(frame)

        # Apply effects based on parameter
        frame = apply_effect(frame, effect)
        # draw_reactangle(frame)
        
        # Encode the processed frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
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
    threading.Thread(target=send_video_to_api, daemon=True).start()
    app.run(host='127.0.0.1', debug=True, port=5001)
