import cv2
import numpy as np

def main():
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(500, 500))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Preprocess the face for the model
            face_gray_resized = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            face_gray_normalized = face_gray_resized / 255.0
            face_gray_reshaped = np.reshape(face_gray_normalized, (1, 48, 48, 1))

        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
