SenseDesktop AI is a real-time multimodal AI desktop copilot designed to transform your daily workstation experience. By combining computer vision, gesture recognition, health sensor analytics, automatic speech recognition (ASR), and Large Language Models (LLM), SenseDesktop AI continuously senses your surrounding environment, guards your posture, and delivers contextual, human-like voice interaction.
✨ Key Features
👁️ Real-time Computer Vision & Object Recognition
YOLOv8 Detection: Real-time object detection and bounding box visualization (laptops, cups, phones, books, etc.).
Heuristic OpenCV Feature Augmentation: Built-in Haar cascade classifiers to recognize personal accessories like glasses even in text-only LLM pipelines.
OCR Text Extraction: Powered by PaddleOCR to capture and extract text directly from books or screens in front of the camera.
💚 Health Monitoring & Fatigue Care
Eye Fatigue & Face-Touching Alerts: Integrates MediaPipe landmarks to detect when hands are touching the upper facial region or rubbing eyes.
Proximity & Distance Guard: Calculates face bounding box ratios to warn users when they get too close to the screen.
Sedentary Warning: Automatically tracks continuous sitting time and triggers break reminders.
🖐️ Gesture-Based Quick Controls
Wave Gesture (wave): Quick-clears chat history and UI logs.
OK Gesture (ok): Replays the latest AI response via text-to-speech.
🎙️ Natural Multimodal Interaction
Speech-to-Text (ASR): Voice input integration for hands-free queries.
Text-to-Speech (TTS): Realistic neural voice output using Microsoft Edge-TTS (zh-CN-XiaoxiaoNeural).
Context-Aware LLM Reasoning: Feeds structured real-time sensor parameters (detected items, clothes color, posture warnings, OCR text) to the LLM for empathetic, witty, and contextual responses.
🏗️ System Architecture
                                +---------------------------+
                                |     Camera Video Stream   |
                                +-------------+-------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
            +--------v--------+                               +--------v--------+
            |  YOLOv8 & OpenCV|                               |    MediaPipe    |
            | Object & Glasses|                               | Hands & Face    |
            +--------+--------+                               +--------+--------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                   +----------v----------+
                                   |  Sensor Context     |
                                   |  Aggregator & OCR   |
                                   +----------+----------+
                                              |
                                   +----------v----------+
                                   | DeepSeek LLM Engine |
                                   +----------+----------+
                                              |
                                   +----------v----------+
                                   |  PyQt5 GUI & TTS    |
                                   +---------------------+
🚀 Quick Start
Prerequisites
Python 3.9+
A webcam and microphone
Windows / macOS / Linux
1. Installation
Clone the repository and install required dependencies:
Bash
下载代码
复制代码
git clone
 https://github.com/your-username/SenseDesktop-AI.git
cd
 SenseDesktop-AI

pip install opencv-python PyQt5 ultralytics mediapipe pygame edge-tts SpeechRecognition openai paddleocr
2. Configuration
Set up your DeepSeek API key in environment variables or directly inside speech_and_llm.py:
Bash
下载代码
复制代码
# Windows (CMD)
set
 DEEPSEEK_API_KEY=sk-your-api-key-here

# Linux / macOS / PowerShell
export DEEPSEEK_API_KEY="sk-your-api-key-here"
3. Run the Application
Launch the desktop copilot GUI:
Bash
下载代码
复制代码
python main.py
📁 Project Structure
.
├── main.py              # Main entry point with PyQt5 GUI layout & event handlers
├── yolo_detector.py     # Vision detection engine (YOLOv8, MediaPipe, OpenCV & OCR)
├── speech_and_llm.py    # LLM service client, Edge-TTS engine & Speech Recognition
├── README.md            # Project documentation
└── requirements.txt     # Python dependency list
🛠️ Built With
GUI Framework: PyQt5
Vision & ML: OpenCV, YOLOv8 (Ultralytics), MediaPipe, PaddleOCR
AI & LLM: DeepSeek API (OpenAI Python SDK)
Audio & Speech: Edge-TTS, Pygame, SpeechRecognition
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
