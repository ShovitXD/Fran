import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE        = 16000
SILENCE_THRESHOLD  = 0.0003
SILENCE_DURATION   = 1.2
MAX_RECORD_SECONDS = 30
MIN_SPEECH_SECONDS = 0.4

# ─────────────────────────────────────────────
# MIC SELECTION
# ─────────────────────────────────────────────
def select_mic() -> int:
    print("\nAvailable microphones:")
    print("─" * 50)

    devices = sd.query_devices()
    input_devs = []

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default_mark = " (system default)" if i == sd.default.device[0] else ""
            print(f"[{len(input_devs)}] {dev['name']}{default_mark}")
            input_devs.append((i, dev["name"]))

    print("─" * 50)

    while True:
        try:
            choice = input(f"Pick a mic [0–{len(input_devs)-1}]: ").strip()
            idx = int(choice)
            if 0 <= idx < len(input_devs):
                device_id, name = input_devs[idx]
                print(f"✓ Using: {name}\n")
                return device_id
            else:
                print(f"Please enter a number between 0 and {len(input_devs)-1}.")
        except ValueError:
            print("Please enter a valid number.")

# ─────────────────────────────────────────────
# RECORD UNTIL SILENCE
# ─────────────────────────────────────────────
def record_until_silence(device_id: int):
    chunk_size = int(SAMPLE_RATE * 0.1)
    max_chunks = int(MAX_RECORD_SECONDS / 0.1)
    silence_chunks_needed = int(SILENCE_DURATION / 0.1)

    audio_chunks = []
    silent_count = 0
    speech_started = False

    print("🎤 Listening... (speak now)")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device_id) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > SILENCE_THRESHOLD:
                audio_chunks.append(chunk)
                silent_count = 0
                speech_started = True
            else:
                if speech_started:
                    audio_chunks.append(chunk)
                    silent_count += 1
                    if silent_count >= silence_chunks_needed:
                        break

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0).flatten()

    if len(audio) / SAMPLE_RATE < MIN_SPEECH_SECONDS:
        return None

    return audio

# ─────────────────────────────────────────────
# TRANSCRIBE
# ─────────────────────────────────────────────
def transcribe(whisper_model, audio: np.ndarray):
    print("⏳ Transcribing...")
    segments, info = whisper_model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        word_timestamps=False
    )
    text = " ".join(segment.text for segment in segments).strip()
    return text

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("=== Fran ASR Test Loop (faster-whisper) ===")
    print("Ctrl+C to exit\n")

    device_id = select_mic()

    print("Loading faster-whisper large-v3-turbo...")
    whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    print("✅ ASR Ready!\n")

    while True:
        try:
            audio = record_until_silence(device_id)
            if audio is None:
                continue

            text = transcribe(whisper_model, audio)

            if text:
                print(f"\n👂 You said: {text}\n")
            else:
                print("No speech detected.\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()