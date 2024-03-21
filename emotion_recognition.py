import cv2

def main():
    capture = cv2.VideoCapture(1)

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
