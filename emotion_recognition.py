import cv2
import numpy as np
import keras
from keras.models import load_model

capture = cv2.VideoCapture(0)
model = load_model('output/model_emotion.keras')
classes = ['angry', 'contempt', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def main():

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                raise IOError("Failed to load haarcascade for face detection.")
        except Exception as e:
            print(e)
            exit(1)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Preprocess the face for the model
            face_gray_resized = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_gray_normalized = face_gray_resized / 255.0
            face_gray_reshaped = np.reshape(face_gray_normalized, (1, 48, 48, 1))

            prediction = model.predict(face_gray_reshaped)
            max_index = np.argmax(prediction)
            predicted_emotion = classes[max_index]
            print(prediction)

            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)


        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
