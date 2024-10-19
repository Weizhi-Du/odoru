from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app)

# Placeholder for frame buffer (60 frames = 2 seconds at 30fps)
frame_buffer = []
batch_size = 60

# MediaPipe pose detection setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_batch(frames):
    scores = []
    for frame in frames:
        img_data = base64.b64decode(frame)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        score = len(results.pose_landmarks.landmark) if results.pose_landmarks else 0
        scores.append(score)

    return sum(scores) // len(scores) if scores else 0

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    global frame_buffer

    frame_buffer.append(data['frame'])

    if len(frame_buffer) >= batch_size:
        batch_frames = frame_buffer[:batch_size]
        frame_buffer = frame_buffer[batch_size:]
        
        score = process_batch(batch_frames)
        emit('batch_score', {'score': score})

if __name__ == '__main__':
    socketio.run(app, debug=True)
