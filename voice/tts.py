import os
import threading
import numpy as np
import sounddevice as sd
import certifi
from dotenv import load_dotenv
from google import genai
from google.genai import types

# fix SSL on Windows
os.environ["SSL_CERT_FILE"] = certifi.where()

SECRETS_PATH = "D:/1- Importants/Notes/fran_secrets.env"
TTS_OUTPUT_DEVICE = 20

_client = None
_lock = threading.Lock()


def init_tts():
    global _client

    print("[Fran] Loading Gemini TTS...")

    load_dotenv(SECRETS_PATH)
    api_key = os.getenv("GOOGLE_API_KEY")

    _client = genai.Client(api_key=api_key)

    print("[Fran] TTS Ready")


def speak(text):
    global _client

    if not text.strip():
        return

    with _lock:
        response = _client.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Leda"
                        )
                    )
                ),
            ),
        )

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        sd.play(audio_array, samplerate=24000, device=TTS_OUTPUT_DEVICE)
        sd.wait()