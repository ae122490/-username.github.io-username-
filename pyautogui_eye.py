import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import pyglet
import matplotlib.pyplot as plt

from time import sleep
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pynput.mouse import Controller, Button
from math import hypot
mouse = Controller()


# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Initialize camera
cap = cv2.VideoCapture(0)

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Add a small pause between PyAutoGUI commands
pyautogui.PAUSE = 0.1


# Load sounds
sound = pyglet.media.load("sound.wav", streaming=False)
left_sound = pyglet.media.load("left.wav", streaming=False)
right_sound = pyglet.media.load("right.wav", streaming=False)

def smooth_mouse_movement(current_x, current_y, target_x, target_y, smoothing_factor=0.2):
    return (
        current_x + (target_x - current_x) * smoothing_factor,
        current_y + (target_y - current_y) * smoothing_factor
    )

def calculate_eye_aspect_ratio(eye_landmarks):
    # Calculate the eye aspect ratio
    vertical_dist1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_dist2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def calculate_mouth_aspect_ratio(mouth_landmarks):
    # Calculate the mouth aspect ratio
    vertical_dist = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
    horizontal_dist = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])
    mar = vertical_dist / horizontal_dist
    return mar

def calibrate_eye_tracker():
    calibration_points = [
        (0.1, 0.1),     (0.5, 0.1),     (0.9, 0.1),
         (0.15, 0.3), (0.3, 0.2),(0.7, 0.2), (0.85, 0.3),
        (0.1, 0.5),     (0.5, 0.5),     (0.9, 0.5),
         (0.15, 0.7), (0.3, 0.8),(0.7, 0.8), (0.85, 0.7), 
        (0.1, 0.9),     (0.5, 0.9),     (0.9, 0.9)
    ]
    
    calibration_data = []
    
    for point in calibration_points:
        screen_x = int(point[0] * screen_w)
        screen_y = int(point[1] * screen_h)

        calibration_image = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(calibration_image, (screen_x, screen_y), 10, (0, 255, 0), -1)
        cv2.imshow('Calibration', calibration_image)

        eye_positions = []
        ret, frame = cap.read()
        if not ret:
            continue
        _, frame = cap.read()
        #frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为 LAB 颜色空间
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2Lab)
        
        # 分离通道
        l, a, b = cv2.split(lab)
        
        # 对亮度通道进行直方图均衡化
        l_eq = cv2.equalizeHist(l)
        
        # 合并均衡化后的通道
        lab_eq = cv2.merge((l_eq, a, b))
        
        # 转换回 BGR 颜色空间
        equalized_image = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
        
        results = face_mesh.process(equalized_image)

        landmark_points = results.multi_face_landmarks
        
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
        time.sleep(500)
        if results.multi_face_landmarks:        
            for _ in range(50):  # Increased from 5 to 50 for more stable calibration
                
                if landmark_points:
                    landmarks = landmark_points[0].landmark
                    frame_h, frame_w, _ = frame.shape
                    
                    left_pupil_x = int(landmarks[468].x * frame_w)
                    left_pupil_y = int(landmarks[468].y * frame_h)
                    
                    right_pupil_x = int(landmarks[473].x * frame_w)
                    right_pupil_y = int(landmarks[473].y * frame_h)
        
                    center_pupil_x = (left_pupil_x + right_pupil_x) // 2
                    center_pupil_y = (left_pupil_y + right_pupil_y) // 2
                    
                    eye_position = (center_pupil_x, center_pupil_y)
                    eye_positions.append(eye_position)
                    
        
        if eye_positions:
            avg_eye_position = np.mean(eye_positions, axis=0)
            calibration_data.append((avg_eye_position, (screen_x, screen_y)))
    
    cv2.destroyWindow('Calibration')

    # Polynomial Regression
    X = np.array([data[0] for data in calibration_data])
    y = np.array([data[1] for data in calibration_data])
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    model_x = LinearRegression().fit(X_poly, y[:, 0])
    model_y = LinearRegression().fit(X_poly, y[:, 1])
    
    return model_x, model_y, poly, X, y

def predict_gaze(gaze_ratio, vertical_gaze, model_x, model_y, poly):
    eye_features = np.array([[gaze_ratio, vertical_gaze]])
    eye_features_poly = poly.transform(eye_features)
    screen_x = model_x.predict(eye_features_poly)[0]
    screen_y = model_y.predict(eye_features_poly)[0]
    return int(screen_x), int(screen_y)

# Perform calibration
print("Starting calibration process, please follow the green dot with your gaze...")
model_x, model_y, poly, X, y = calibrate_eye_tracker()
print("Calibration complete!")

# Initialize current mouse position
current_mouse_x, current_mouse_y = pyautogui.position()

# 初始化變量
blink_threshold = 0.5  # 眨眼閾值
mouth_threshold = 0.72  # 嘴巴開合閾值
squint_threshold = 0.25  # 瞇眼閾值
eye_control_enabled = False

# 滑鼠控制模式
mouse_control_enabled = False
last_mar_state = False  # 用來追踪嘴巴狀態
blinking_frames = 0  # 用來追踪眨眼框架數
frames_to_blink = 15  # 定義眨眼需要持續的框架數
frames_to_open = 40  # 定義眨眼需要持續的框架數

# EAR 和 MAR 閾值
EAR_THRESHOLD = 0.19  # 眨眼閾值
MAR_THRESHOLD = 0.85   # 嘴巴開合閾值

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 转换为 LAB 颜色空间
    lab = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2Lab)
    
    # 分离通道
    l, a, b = cv2.split(lab)
    
    # 对亮度通道进行直方图均衡化
    l_eq = cv2.equalizeHist(l)
    
    # 合并均衡化后的通道
    lab_eq = cv2.merge((l_eq, a, b))
    
    # 转换回 BGR 颜色空间
    equalized_image = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
    
    results = face_mesh.process(equalized_image)
    rows, cols, _ = frame.shape
    # 畫一個白色條用來顯示眨眼進度
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
        #print(landmarks)
        # 眼睛和嘴巴的關鍵點
        left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
        right_eye = landmarks[[362, 385, 387, 263, 373, 380]]
        mouth = landmarks[[61, 291, 39, 181, 0, 17, 269, 405]]
        
        # 計算眼睛和嘴巴的縱橫比
        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mouth_aspect_ratio(mouth)

        # 雙眼開啟
        if (left_ear<EAR_THRESHOLD) and (right_ear<EAR_THRESHOLD):
            blinking_frames += 1
            percentage_blinking = blinking_frames / frames_to_open
            loading_x = int(cols * percentage_blinking)
            cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
            if blinking_frames >= frames_to_open:
                #cv2.putText(frame, "BLINKING", (50, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=3)
                sound.play()
                eye_control_enabled = not eye_control_enabled
                #cv2.waitKey(1)                
                print(f"Mouse control {'enabled' if eye_control_enabled else 'disabled'}")                 
                blinking_frames = 0  # 重置眨眼偵測
        else :
            # 檢測眨眼
            if eye_control_enabled:
                if left_ear<EAR_THRESHOLD:
                    blinking_frames += 1
                    percentage_blinking = blinking_frames / frames_to_blink
                    loading_x = int(cols * percentage_blinking)
                    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
                    if blinking_frames >= frames_to_blink:
                        #cv2.putText(frame, "BLINKING", (50, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=3)
                        mouse.click(Button.left)
                        left_sound.play()
                        blinking_frames = 0  # 重置眨眼偵測
                        cv2.putText(frame, " left blinking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("左眼眨眼偵測: 左鍵點擊") 
                elif right_ear<EAR_THRESHOLD:
                    blinking_frames += 1
                    percentage_blinking = blinking_frames / frames_to_blink
                    loading_x = int(cols * percentage_blinking)
                    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)
                    if blinking_frames >= frames_to_blink:
                        #cv2.putText(frame, "BLINKING", (50, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), thickness=3)
                        mouse.click(Button.right)
                        right_sound.play()
                        blinking_frames = 0  # 重置眨眼偵測
                        cv2.putText(frame, " right blinking", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print("右眼眨眼偵測: 右鍵點擊")
                else:
                    # 眼動追蹤和滑鼠控制
                    if eye_control_enabled:
                        
                        left_pupil = landmarks[468]
                        right_pupil = landmarks[473]
                        center_pupil_x = int((left_pupil[0] + right_pupil[0]) / 2 * frame.shape[1])
                        center_pupil_y = int((left_pupil[1] + right_pupil[1]) / 2 * frame.shape[0])

                        gaze_x, gaze_y = predict_gaze(center_pupil_x, center_pupil_y, model_x, model_y, poly)
                        gaze_x = max(0, min(gaze_x, screen_w))
                        gaze_y = max(0, min(gaze_y, screen_h))
                        
                        current_mouse_x, current_mouse_y = smooth_mouse_movement(
                            current_mouse_x, current_mouse_y, gaze_x, gaze_y
                        )
                        
                        try:
                            pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y))
                        except Exception as e:
                            print(f"Error moving mouse: {e}")
            
            else:
                blinking_frames = 0  # 重置眨眼偵測
                
        
        # 瞇眼
        '''if avg_ear < squint_threshold:
            pyautogui.scroll(-1)  # 向下滾動
        elif avg_ear > blink_threshold and avg_ear < (blink_threshold + squint_threshold) / 2:
            pyautogui.scroll(1)  # 向上滾動'''
        

        
        # 繪製視覺反饋
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Eye Control: {'On' if eye_control_enabled else 'Off'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    

    
    cv2.imshow("Enhanced Eye-Tracking System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()