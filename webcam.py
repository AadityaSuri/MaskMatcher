# citation: https://github.com/shantnu/Webcam-Face-Detect
#           https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2

cascPath = "haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0) # live webcam feed object

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # converts image to grayscale
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #draw rectangles around detected object
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #show webcam stream
    cv2.imshow('Video', frame)

    #exit condition
    if cv2.waitKey(1) & 0xFF == ord('E'):
        break

    cv2.imshow('Video', frame)
video_capture.release()
cv2.destroyAllWindows()

