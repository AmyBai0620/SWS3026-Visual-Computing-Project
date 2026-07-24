import cv2
import time
import csv
import numpy as np
import mediapipe as mp

# ========== 实验场景列表(可自由增删) ==========
SCENARIOS = [
    'turn_left_30',          # 左转约30度
    'turn_left_60',          # 左转约60度
    'head_tilt',             # 歪头:左右歪约45度(绕视线轴滚转)
    'look_up_down',          # 抬头低头
    'cover_mouth',           # 手遮嘴
    'cover_one_eye',         # 手遮一只眼
    'dim_light',             # 暗光(关灯再按空格)
    'backlight',             # 逆光(背对窗户)
    'close_up',              # 凑近
    'far_away',              # 退远1.5-2米
]
CAPTURE_SECONDS = 10         # 每个场景采集时长(秒)
DOT_RADIUS = 1.8             # MediaPipe 关键点半径(支持小数)
DOT_R16 = int(round(DOT_RADIUS * 16))   # 转成 1/16 像素定点

# ========== 初始化检测器 ==========
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel('lbfmodel.yaml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ========== 状态变量 ==========
scenario_idx = 0             # 当前是第几个场景
recording = False            # 是否在采集中
record_start = 0             # 本场景采集开始时刻
stats = None                 # 本场景的统计数据
results = []                 # 所有场景的最终结果

def new_stats():
    return {'haar_hit': 0, 'haar_ms': 0.0,
            'mp_hit': 0, 'mp_ms': 0.0, 'frames': 0}

cap = cv2.VideoCapture(0)
cv2.namedWindow('Auto Experiment', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Auto Experiment', 1280, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    left, right = frame.copy(), frame.copy()

    # ---------- Haar + LBF ----------
    t0 = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    haar_detected = len(faces) > 0
    if haar_detected:
        ok, landmarks = facemark.fit(gray, faces)
        if ok:
            for lm in landmarks:
                for (px, py) in lm.reshape(-1, 2):
                    cv2.circle(left, (int(px), int(py)), 2, (0, 0, 255), -1)
        for (x, y, w, h) in faces:
            cv2.rectangle(left, (x, y), (x + w, y + h), (0, 255, 0), 2)
    haar_ms = (time.time() - t0) * 1000

    # ---------- MediaPipe ----------
    t0 = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    mp_detected = result.multi_face_landmarks is not None
    if mp_detected:
        h, w = frame.shape[:2]
        for face_lms in result.multi_face_landmarks:
            for p in face_lms.landmark:
                # shift=4 表示坐标/半径按 1/16 像素定点,这样半径能取到小数
                cv2.circle(right, (int(p.x * w * 16), int(p.y * h * 16)),
                           DOT_R16, (255, 0, 0), -1, cv2.LINE_AA, 4)
    mp_ms = (time.time() - t0) * 1000

    combined = np.hstack([left, right])

    # ---------- 每个 panel 的检测状态(让"失败"也看得见) ----------
    ph, pw = frame.shape[:2]
    for x0, name, hit, ms in ((0, 'Haar+LBF', haar_detected, haar_ms),
                              (pw, 'MediaPipe', mp_detected, mp_ms)):
        color = (0, 255, 0) if hit else (0, 0, 255)
        tag = 'DETECTED' if hit else 'NO FACE'
        cv2.putText(combined, f'{name}: {tag}  {ms:.0f}ms',
                    (x0 + 10, ph - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # ---------- 状态机逻辑 ----------
    if scenario_idx >= len(SCENARIOS):
        # 全部完成
        cv2.putText(combined, 'ALL DONE! Results saved. Press q to exit.',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    elif not recording:
        # 等待开始:显示当前场景名,提示按空格
        name = SCENARIOS[scenario_idx]
        cv2.putText(combined, f'[{scenario_idx+1}/{len(SCENARIOS)}] Next: {name}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(combined, 'Get ready, press SPACE to record 10s',
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        # 采集中:累计统计 + 显示倒计时
        stats['frames'] += 1
        stats['haar_hit'] += 1 if haar_detected else 0
        stats['haar_ms'] += haar_ms
        stats['mp_hit'] += 1 if mp_detected else 0
        stats['mp_ms'] += mp_ms

        elapsed = time.time() - record_start
        remain = CAPTURE_SECONDS - elapsed
        cv2.putText(combined, f'RECORDING {SCENARIOS[scenario_idx]}  {remain:.1f}s',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if elapsed >= CAPTURE_SECONDS:
            # 本场景结束:算结果、存截图、进入下一场景
            f = stats['frames']
            results.append({
                'scenario': SCENARIOS[scenario_idx],
                'frames': f,
                'haar_hit_rate_%': round(stats['haar_hit'] / f * 100, 1),
                'mp_hit_rate_%': round(stats['mp_hit'] / f * 100, 1),
                'haar_avg_ms': round(stats['haar_ms'] / f, 1),
                'mp_avg_ms': round(stats['mp_ms'] / f, 1),
            })
            cv2.imwrite(f'shot_{SCENARIOS[scenario_idx]}.png', combined)
            print(f'完成: {results[-1]}')
            scenario_idx += 1
            recording = False

    cv2.imshow('Auto Experiment', combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32 and not recording and scenario_idx < len(SCENARIOS):  # 空格=32
        recording = True
        record_start = time.time()
        stats = new_stats()

# ---------- 写出 CSV ----------
if results:
    with open('results.csv', 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\n已保存 {len(results)} 个场景的数据到 results.csv')

cap.release()
cv2.destroyAllWindows()