import cv2
import time
import os
import numpy as np
from ultralytics import YOLO

# 兼容 MediaPipe 导入
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# 尝试加载 PaddleOCR
HAS_OCR = False
try:
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    HAS_OCR = True
except Exception:
    HAS_OCR = False


class YoloDetector:
    def __init__(self, model_path="yolov8s.pt"):
        print(f"[YOLO] 正在加载模型: {model_path}...")
        self.model = YOLO(model_path)
        self.current_objects = []
        self.clothes_color = "灰色"
        self.ocr_text = ""
        self.health_warning = ""

        # 加载 OpenCV 自带的眼睛/眼镜 Haar 级联检测器
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 健康监控计数与计时
        self.sit_start_time = time.time()  # 久坐计时起始
        self.last_ocr_time = time.time()   # OCR 频率控制

        self.has_mp = HAS_MEDIAPIPE
        if self.has_mp:
            try:
                self.mp_hands = mp_hands
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
                self.mp_draw = mp_draw
                print("[MediaPipe] 健康、姿态与距屏监测已激活！")
            except Exception:
                self.has_mp = False

    def _get_dominant_color(self, crop_img):
        if crop_img is None or crop_img.size == 0:
            return "灰色"

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        color_ranges = {
            "白色": (np.array([0, 0, 180]), np.array([180, 45, 255])),
            "黑色": (np.array([0, 0, 0]), np.array([180, 255, 50])),
            "灰色": (np.array([0, 0, 50]), np.array([180, 45, 180])),
            "红色": (np.array([0, 100, 100]), np.array([10, 255, 255])),
            "蓝色": (np.array([100, 100, 100]), np.array([130, 255, 255])),
            "绿色": (np.array([35, 100, 100]), np.array([85, 255, 255])),
            "黄色": (np.array([20, 100, 100]), np.array([35, 255, 255])),
        }

        max_pixels = 0
        dominant_color = "灰色"
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            count = cv2.countNonZero(mask)
            if count > max_pixels:
                max_pixels = count
                dominant_color = color_name

        return dominant_color

    def _detect_glasses(self, frame):
        """利用 OpenCV 特征检测用户是否佩戴眼镜"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        has_glasses = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + int(h*0.6), x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            if len(eyes) >= 1:
                has_glasses = True
                break
        return has_glasses

    def _recognize_gesture(self, hand_landmarks):
        lm = hand_landmarks.landmark
        middle_open = lm[12].y < lm[10].y
        ring_open = lm[16].y < lm[14].y
        pinky_open = lm[20].y < lm[18].y
        thumb_index_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)**0.5

        if thumb_index_dist < 0.05 and middle_open and ring_open:
            return "ok"
        elif lm[8].y < lm[6].y and middle_open and ring_open and pinky_open:
            return "wave"
        return None

    def _run_ocr(self, frame):
        """利用 PaddleOCR 识别文字"""
        if not HAS_OCR:
            return ""
        try:
            result = ocr_engine.ocr(frame, cls=True)
            text_list = []
            if result and result[0]:
                for line in result[0]:
                    text_list.append(line[1][0])
            return " ".join(text_list)
        except Exception:
            return ""

    def detect(self, frame):
        self.health_warning = ""

        # 1. YOLO 检测与标签转换
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()

        objects_detected = []
        label_map = {
            "bottle": "水瓶/水杯", "cup": "水杯", "cell phone": "手机",
            "book": "书本", "laptop": "笔记本电脑", "mouse": "鼠标",
            "keyboard": "键盘", "person": "人", "scissors": "剪刀", "pen": "笔"
        }

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw_label = self.model.names[cls_id]
            objects_detected.append(label_map.get(raw_label, raw_label))

            if raw_label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_area = frame[int(y1+(y2-y1)*0.4):int(y1+(y2-y1)*0.8), int(x1+(x2-x1)*0.2):int(x1+(x2-x1)*0.8)]
                self.clothes_color = self._get_dominant_color(crop_area)

        # 2. 眼镜算法辅助补充
        if self._detect_glasses(frame):
            objects_detected.append("眼镜")

        self.current_objects = list(set(objects_detected))

        # 3. 健康监测：久坐提醒 (> 45 分钟)
        sit_duration = (time.time() - self.sit_start_time) / 60
        if "人" in self.current_objects and sit_duration > 45:
            self.health_warning += "【久坐预警】您已连续工作超过 45 分钟；"

        # 4. MediaPipe 人脸与手势/揉眼/摸脸监测
        gesture = None
        if self.has_mp:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # (1) 获取人脸 Bounding Box
            face_results = self.face_detection.process(rgb_frame)
            face_boxes = []
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    face_ratio = bbox.width * bbox.height
                    if face_ratio > 0.18:
                        self.health_warning += "【视力保护】眼睛离屏幕过近；"
                    
                    # 记录人脸坐标 (xmin, ymin, xmax, ymax)
                    face_boxes.append((
                        bbox.xmin,
                        bbox.ymin,
                        bbox.xmin + bbox.width,
                        bbox.ymin + bbox.height
                    ))

            # (2) 手势与揉眼/摸脸精准交叉判断
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self._recognize_gesture(hand_landmarks)

                    # 提取关键点：食指尖(8)、中指尖(12)、手掌基部(0)
                    index_x = hand_landmarks.landmark[8].x
                    index_y = hand_landmarks.landmark[8].y
                    middle_x = hand_landmarks.landmark[12].x
                    middle_y = hand_landmarks.landmark[12].y

                    # 判断手部是否与人脸区域重叠（特别关注人脸上半部分/眼睛）
                    touch_face = False
                    if face_boxes:
                        for (fx1, fy1, fx2, fy2) in face_boxes:
                            # 扩大一点人脸判断范围，包含眼周
                            margin = 0.05
                            if (fx1 - margin <= index_x <= fx2 + margin and fy1 - margin <= index_y <= fy2 + margin) or \
                               (fx1 - margin <= middle_x <= fx2 + margin and fy1 - margin <= middle_y <= fy2 + margin):
                                touch_face = True
                                break
                    else:
                        # 若未获取到 Face Box，采用全局比例防护降级 (y < 0.6)
                        if index_y < 0.60 or middle_y < 0.60:
                            touch_face = True

                    if touch_face:
                        self.health_warning += "【疲劳预警】正在揉眼睛或频繁摸脸；"

        # 5. PaddleOCR 定时触发 (每 3 秒)
        if time.time() - self.last_ocr_time > 3.0:
            self.ocr_text = self._run_ocr(frame)
            self.last_ocr_time = time.time()

        detect_result = {
            "gesture": gesture,
            "objects": self.current_objects,
            "clothes_color": self.clothes_color,
            "ocr_text": self.ocr_text,
            "health_warning": self.health_warning
        }

        return annotated_frame, detect_result
