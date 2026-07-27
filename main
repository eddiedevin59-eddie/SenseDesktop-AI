import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch_lib_path = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_path):
    try:
        os.add_dll_directory(torch_lib_path)
    except AttributeError:
        pass

import time
import threading
import cv2

from yolo_detector import YoloDetector
from speech_and_llm import LLMService, TTSService, ASRService

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap


class WorkerSignals(QObject):
    update_chat_signal = pyqtSignal(str, str)


class SmartCopilotWindow(QWidget):
    def __init__(self, yolo_detector, llm_service, tts_service, asr_service, signals):
        super().__init__()
        self.detector = yolo_detector
        self.llm = llm_service
        self.tts = tts_service
        self.asr = asr_service
        self.signals = signals

        self.cap = cv2.VideoCapture(0)
        self.latest_detect_result = None
        self.last_gesture_time = time.time()

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def init_ui(self):
        self.setWindowTitle("AI Copilot 智能桌面助手 (全模态关怀版)")
        self.resize(1180, 720)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧：感知监控窗口
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.video_label = QLabel("正在初始化摄像头...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: #ffffff; border-radius: 8px;")
        self.video_label.setMinimumSize(520, 390)

        self.gesture_status = QLabel("🖐️ 手势控制: [ 👋 已清空聊天记录 ]")
        self.gesture_status.setFont(QFont("Microsoft YaHei", 9))
        self.gesture_status.setStyleSheet("color: #666; margin-top: 5px;")

        self.health_status = QLabel("💚 健康监测: 姿态良好，距离正常")
        self.health_status.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self.health_status.setStyleSheet("color: #2b8a3e; margin-top: 2px;")

        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.gesture_status)
        left_layout.addWidget(self.health_status)
        splitter.addWidget(left_widget)

        # 右侧：对话与交互
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.status_label = QLabel("🎤 语音与多模态助手: 随时倾听中")
        self.status_label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.status_label.setStyleSheet("color: #1c7ed6;")
        right_layout.addWidget(self.status_label)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 10))
        self.chat_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #ced4da; border-radius: 5px; padding: 10px;")
        right_layout.addWidget(self.chat_display)

        # 输入栏
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入或点击【🎤 语音输入】（例如：'我戴眼镜了吗'、'1+1等于几'）...")
        self.input_field.setFont(QFont("Microsoft YaHei", 10))
        self.input_field.setStyleSheet("padding: 8px; border: 1px solid #ced4da; border-radius: 5px;")
        self.input_field.returnPressed.connect(self.send_message)

        self.voice_button = QPushButton("🎤 语音输入")
        self.voice_button.setFont(QFont("Microsoft YaHei", 10))
        self.voice_button.setStyleSheet("background-color: #40c057; color: white; padding: 8px 12px; border-radius: 5px;")
        self.voice_button.clicked.connect(self.listen_voice)

        self.send_button = QPushButton("发送")
        self.send_button.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.send_button.setStyleSheet("background-color: #1c7ed6; color: white; padding: 8px 18px; border-radius: 5px;")
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.voice_button)
        input_layout.addWidget(self.send_button)
        right_layout.addLayout(input_layout)

        splitter.addWidget(right_widget)
        splitter.setSizes([580, 600])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        detect_result = None
        if hasattr(self.detector, 'detect'):
            result = self.detector.detect(frame)
            if isinstance(result, tuple):
                frame, detect_result = result
            else:
                frame = result

        self.latest_detect_result = detect_result

        # 更新健康状态与提醒
        if detect_result and detect_result.get("health_warning"):
            self.health_status.setText(f"⚠️ 健康提醒: {detect_result.get('health_warning')}")
            self.health_status.setStyleSheet("color: #e03131; font-weight: bold;")
        else:
            self.health_status.setText("💚 健康监测: 姿态良好，距离正常")
            self.health_status.setStyleSheet("color: #2b8a3e; font-weight: bold;")

        # 检查手势控制
        self._check_gesture(detect_result)

        # 画面渲染
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _check_gesture(self, detect_result):
        if not isinstance(detect_result, dict):
            return

        current_time = time.time()
        gesture = detect_result.get("gesture", None)

        if gesture and (current_time - self.last_gesture_time > 1.5):
            if gesture == "wave":
                self.chat_display.clear()
                self.gesture_status.setText("🖐️ 手势控制: [ 👋 已清空聊天记录 ]")
                self.last_gesture_time = current_time
            elif gesture == "ok":
                text_content = self.chat_display.toPlainText().strip()
                if text_content:
                    lines = [l for l in text_content.split('\n') if l.strip()]
                    if lines:
                        if hasattr(self.tts, 'speak'):
                            self.tts.speak(lines[-1])
                self.last_gesture_time = current_time

    def append_chat(self, sender, message):
        color = "#1971c2" if sender == "User" else "#2b8a3e"
        self.chat_display.append(f"<span style='color:{color}; font-weight:bold;'>[{sender}]:</span> {message}")
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def listen_voice(self):
        def _asr_thread():
            self.signals.update_chat_signal.emit("System", "正在倾听中，请说话...")
            recognized_text = self.asr.listen_and_recognize()
            if recognized_text:
                self.signals.update_chat_signal.emit("User (语音)", recognized_text)
                self._process_llm_request(recognized_text)
            else:
                self.signals.update_chat_signal.emit("System", "未能清晰识别语音，请重试。")

        threading.Thread(target=_asr_thread, daemon=True).start()

    def send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return

        self.append_chat("User", text)
        self.input_field.clear()
        threading.Thread(target=self._process_llm_request, args=(text,), daemon=True).start()

    def _process_llm_request(self, user_text):
        try:
            current_objects = getattr(self.detector, 'current_objects', [])
            clothes_color = getattr(self.detector, 'clothes_color', '灰色')
            ocr_text = getattr(self.detector, 'ocr_text', '')
            health_warning = getattr(self.detector, 'health_warning', '')

            if hasattr(self.llm, 'get_response'):
                response = self.llm.get_response(
                    user_text,
                    detected_objects=current_objects,
                    clothes_color=clothes_color,
                    ocr_text=ocr_text,
                    health_warning=health_warning
                )
            else:
                response = "未能找到有效的 LLM 响应接口。"

            self.signals.update_chat_signal.emit("AI Copilot", str(response))

            if hasattr(self.tts, 'speak'):
                self.tts.speak(str(response))

        except Exception as e:
            self.signals.update_chat_signal.emit("System", f"处理出错: {e}")

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    signals = WorkerSignals()

    print("[System] 初始化纯文本 Sensor 融合与 DeepSeek 桌面助手...")
    llm_service = LLMService()
    tts_service = TTSService()
    asr_service = ASRService()
    yolo_detector = YoloDetector()

    window = SmartCopilotWindow(yolo_detector, llm_service, tts_service, asr_service, signals)
    window.show()

    signals.update_chat_signal.connect(window.append_chat)
    sys.exit(app.exec_())
