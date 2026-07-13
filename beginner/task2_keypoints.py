import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# create LBF keypoint detector & load model
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')  
# Open webcam
cap = cv2.VideoCapture(0)

cv2.namedWindow('Face Keypoints', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Keypoints', 640, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # image shrink scale
        minNeighbors=5,     # min detection neighbors
        minSize=(60, 60)    # min face size
    )

    # draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # image leftup, rightdown, colorBGR, line width

    # predict and draw keypoints
    if len(faces) > 0:                       
        ok, landmarks = facemark.fit(gray, faces)
        if ok:
            for lm in landmarks:        
                points = lm.reshape(-1, 2)        
                for (px, py) in points:   
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1) # image center radius  color  -1(fill)

    cv2.imshow('Face Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()