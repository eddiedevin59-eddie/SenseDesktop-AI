import os
import asyncio
import tempfile
import threading
import pygame
from openai import OpenAI

# 初始化 pygame 播放器用于 Edge-TTS 播报
pygame.mixer.init()

class LLMService:
    def __init__(self):
        # ⚠️ 请确认填入你真实的 DeepSeek API Key
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "sk       ")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_response(self, prompt, frame=None, detected_objects=None, clothes_color="灰色", ocr_text="", health_warning=""):
        if detected_objects is None:
            detected_objects = []

        holding_candidates = [obj for obj in detected_objects if obj not in ["人", "笔记本电脑", "laptop", "person"]]
        objects_str = ", ".join(detected_objects) if detected_objects else "未发现明确物体"
        holding_str = ", ".join(holding_candidates) if holding_candidates else "未识别到突出物品"
        ocr_str = ocr_text if ocr_text else "画面中无明显文字"

        has_glasses_str = "佩戴了眼镜" if "眼镜" in detected_objects else "未检测到佩戴眼镜"

        system_prompt = (
            f"你是一个桌面智能 Copilot 助手，语气幽默、自然、接地气、体贴。\n"
            f"【实时传感器与视觉检测数据】：\n"
            f"- 全图识别到的目标/配件：[{objects_str}]\n"
            f"- 眼镜佩戴状态：【{has_glasses_str}】\n"
            f"- 手部可能拿持物品：[{holding_str}]\n"
            f"- 镜头识别到的文字内容（OCR）：【{ocr_str}】\n"
            f"- 用户衣服主色调：【{clothes_color}】\n"
            f"- 当前姿态与健康异常警告：【{health_warning if health_warning else '健康状况良好'}】\n\n"
            f"【回复规则】：\n"
            f"1. 请根据上述最新的传感器及视觉数据回答用户的问题（例如用户问是否戴眼镜，直接依据【眼镜佩戴状态】回答）。\n"
            f"2. 如果有健康警告（如揉眼睛/距离太近/久坐），请幽默且体贴地提醒用户注意休息。\n"
            f"3. 语言保持诙谐自然，简明扼要，避免机械回答。"
        )

        try:
            # 纯文本模式：支持 DeepSeek 格式
            response = self.client.chat.completions.create(
                model="deepseek-chat",  
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[LLM Error] {e}")
            return f"思考失败: {e}"


class TTSService:
    """使用微软 Edge-TTS 实现带情感的真人播报"""
    def __init__(self, voice="zh-CN-XiaoxiaoNeural"):
        self.voice = voice

    def speak(self, text):
        if not text:
            return
        threading.Thread(target=self._play_tts, args=(text,), daemon=True).start()

    def _play_tts(self, text):
        try:
            import edge_tts
            
            async def _generate():
                communicate = edge_tts.Communicate(text, self.voice)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    tmp_path = f.name
                await communicate.save(tmp_path)
                return tmp_path

            tmp_mp3 = asyncio.run(_generate())

            pygame.mixer.music.load(tmp_mp3)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            pygame.mixer.music.unload()
            os.remove(tmp_mp3)
        except Exception as e:
            print(f"[Edge-TTS 播报]: {text} (播报失败: {e})")


class ASRService:
    """语音识别服务 (结合 SpeechRecognition 库)"""
    def listen_and_recognize(self):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("[ASR] 正在倾听...")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
            text = r.recognize_google(audio, language="zh-CN")
            print(f"[ASR 识别结果]: {text}")
            return text
        except Exception as e:
            print(f"[ASR 提示]: 语音未接收或出错 ({e})")
            return ""
