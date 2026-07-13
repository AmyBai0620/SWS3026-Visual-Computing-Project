import cv2
import time
import numpy as np
import mediapipe as mp

# initialize
# 1) Haar + LBF
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

# 2) MediaPipe Face Mesh(468)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


haar_total, haar_hit = 0, 0        # 总帧数 / 检测到脸的帧数
mp_total, mp_hit = 0, 0     
shot_count = 0                   

cap = cv2.VideoCapture(0)
cv2.namedWindow('Haar vs MediaPipe', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Haar vs MediaPipe', 1280, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    left = frame.copy()
    right = frame.copy()  

    # left:Haar + LBF
    t0 = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        ok, landmarks = facemark.fit(gray, faces)
        if ok:
            for lm in landmarks:
                for (px, py) in lm.reshape(-1, 2):
                    cv2.circle(left, (int(px), int(py)), 2, (0, 0, 255), -1)
        for (x, y, w, h) in faces:
            cv2.rectangle(left, (x, y), (x + w, y + h), (0, 255, 0), 2)
    haar_ms = (time.time() - t0) * 1000 # milliseconds
    haar_total += 1
    haar_hit += 1 if len(faces) > 0 else 0

    # right:MediaPipe
    t0 = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # RGB
    result = face_mesh.process(rgb)
    detected = result.multi_face_landmarks is not None
    if detected:
        h, w = frame.shape[:2]
        for face_lms in result.multi_face_landmarks:
            for p in face_lms.landmark:   
                cv2.circle(right, (int(p.x * w), int(p.y * h)), 1, (255, 0, 0), -1)
    mp_ms = (time.time() - t0) * 1000
    mp_total += 1
    mp_hit += 1 if detected else 0


    haar_rate = haar_hit / haar_total * 100
    mp_rate = mp_hit / mp_total * 100
    cv2.putText(left, f'Haar+LBF  {haar_ms:.0f}ms  hit {haar_rate:.0f}%',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(right, f'MediaPipe  {mp_ms:.0f}ms  hit {mp_rate:.0f}%',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


    combined = np.hstack([left, right])
    cv2.imshow('Haar vs MediaPipe', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):        
        shot_count += 1
        cv2.imwrite(f'compare_{shot_count}.png', combined)
        print(f'Saved compare_{shot_count}.png')
    elif key == ord('r'):   # reset statistics
        haar_total = haar_hit = mp_total = mp_hit = 0
        print('Statistics reset.')

cap.release()
cv2.destroyAllWindows()