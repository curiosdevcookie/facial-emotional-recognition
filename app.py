from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app)

# Load your model here
model = load_model('output/model_emotion.keras')
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
THRESHOLD = 0.3

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    try:

        if data.startswith('data:image/jpeg;base64,'):
            data = data[23:]

        image_data = base64.b64decode(data)

        # Convert the image data to a numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            print("Error: Decoded frame is empty")
            return

        # Perform emotion recognition on the frame from the webcam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Failed to load haarcascade for face detection.")
            return
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(300, 300))

        for (x, y, w, h) in faces:
            face_gray_resized = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_gray_normalized = face_gray_resized / 255.0
            face_gray_reshaped = np.reshape(face_gray_normalized, (1, 48, 48, 1))

            prediction = model.predict(face_gray_reshaped)
            max_probability = np.max(prediction)
            max_index = np.argmax(prediction)
            predicted_emotion = classes[max_index]

            print("Max probability:", max_probability)

            if max_probability >= THRESHOLD:
                print("Emitting result: ", {'emotion': predicted_emotion})
                emit('result', {'emotion': predicted_emotion})           
            else:
                emit('result', {'emotion': "neutral"})

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == '__main__':
    socketio.run(app)
