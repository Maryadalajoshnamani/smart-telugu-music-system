
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import random

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json['image']

    # Convert image
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    contrast = gray.std()

    if contrast > 60:
        emotion = "happy"
    elif brightness < 70:
        emotion = "sad"
    else:
        emotion = random.choice(["happy", "sad", "calm"])

    # Telugu songs
    if emotion == "happy":
        search = "telugu mass songs"
    elif emotion == "sad":
        search = "sad telugu songs"
    else:
        search = "telugu instrumental music"

    return jsonify({"emotion": emotion, "search": search})

if __name__ == '__main__':
    app.run(debug=True)