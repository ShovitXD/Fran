import sys
import threading
import queue
import time
import json
import asyncio

import numpy as np
import sounddevice as sd
from pynput import keyboard

from engines.whisper_loader import load_whisper, transcribe
from engines.llm_loader import load_llm, generate
from voice.tts import init_tts, speak
from twitch.twitch_listener import run_twitch_events


SAMPLE_RATE = 16000

BASE_MODEL_PATH = "D:/Fran/brain/base_model"
SYSTEM_PROMPT_PATH = "D:/Fran/brain/system_prompt_runtime.txt"
DATASET_PATH = "D:/Fran/brain/data/Fran_Live_dataset.jsonl"

MAX_NEW_TOKENS = 200

INPUT_QUEUE = queue.Queue()
HISTORY = []
running = True
SYSTEM_PROMPT = ""
recording = False
IS_BUSY = False

last_save_time = 0
SAVE_DEBOUNCE_SECONDS = 1.0


def select_mic():
    print("\nAvailable microphones:")
    print("-" * 50)

    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"[{len(input_devices)}] {dev['name']}")
            input_devices.append(i)

    print("-" * 50)

    while True:
        try:
            choice = int(input("Select mic index: ").strip())

            if 0 <= choice < len(input_devices):
                device_id = input_devices[choice]
                print(f"[Fran] Mic selected: {devices[device_id]['name']}\n")
                return device_id
            else:
                print("[Fran] Invalid choice, try again.")
        except ValueError:
            print("[Fran] Invalid choice, try again.")


def load_all_models():
    global SYSTEM_PROMPT

    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()

    print(f"[Fran] Loaded system prompt ({len(SYSTEM_PROMPT)} chars)")

    load_whisper()
    load_llm(BASE_MODEL_PATH)
    init_tts()

    print("[Fran] Ready.")


def clean(text):
    text = text.strip().lower()
    text = text.rstrip(".!?")

    junk = [
        "thank you",
        "thanks",
        "thank u",
        "thx"
    ]

    if text in junk:
        return None

    if len(text.split()) <= 1:
        return None

    return text


def save_last_exchange():
    global last_save_time

    now = time.time()
    if now - last_save_time < SAVE_DEBOUNCE_SECONDS:
        return
    last_save_time = now

    if len(HISTORY) < 2:
        print("[Fran] Nothing to save.")
        return

    user_msg = HISTORY[-2]["content"]
    assistant_msg = HISTORY[-1]["content"]

    sample = {
        "messages": [
            {
                "role": "user",
                "content": user_msg
            },
            {
                "role": "assistant",
                "content": assistant_msg
            }
        ]
    }

    try:
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print("[Fran] Saved last exchange.")
        print(f"User: {user_msg}")
        print(f"Fran: {assistant_msg}")

    except Exception as e:
        print(f"[Fran] Save failed: {e}")


def on_press(key):
    global recording

    if key == keyboard.Key.down:
        if not recording:
            recording = True
            print("\n[Fran] Recording...")

    elif key == keyboard.Key.left:
        save_last_exchange()


def on_release(key):
    global recording

    if key == keyboard.Key.down:
        recording = False


def audio_loop(device_id):
    global recording

    chunk_size = int(SAMPLE_RATE * 0.1)

    while running:

        if not recording:
            time.sleep(0.01)
            continue

        audio_chunks = []

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_id,
        ) as stream:

            while recording:
                chunk, _ = stream.read(chunk_size)
                audio_chunks.append(chunk)

        if audio_chunks:
            audio = np.concatenate(audio_chunks).flatten()

            text = transcribe(audio)
            text = clean(text)

            if text:
                print(f"\n[PTT]: {text}")
                INPUT_QUEUE.put(f"[Creator]: {text}")

        time.sleep(0.01)


def processor():
    global HISTORY, IS_BUSY

    while running:
        text = INPUT_QUEUE.get()

        if IS_BUSY and text.startswith("[Chat]"):
            print(f"[Fran] Busy, skipping: {text}")
            continue

        IS_BUSY = True

        try:
            print(f"\nInput: {text}")

            t0 = time.perf_counter()

            response = generate(
                SYSTEM_PROMPT,
                HISTORY,
                text,
                MAX_NEW_TOKENS
            )

            print(f"[DEBUG] generate() returned: {repr(response)}")

            t1 = time.perf_counter()

            print(f"\nFran: {response}")
            print(f"[Latency] {t1 - t0:.2f} sec\n")

            try:
                speak(response)
            except Exception as tts_err:
                print(f"[Fran] TTS error: {tts_err}")

            HISTORY.append({"role": "user", "content": text})
            HISTORY.append({"role": "assistant", "content": response})
            HISTORY[:] = HISTORY[-24:]

        except Exception as e:
            print(f"[Fran] Processor error: {e}")

        finally:
            IS_BUSY = False


def twitch_thread():
    asyncio.run(run_twitch_events(INPUT_QUEUE))


def main():
    global running

    print("=" * 60)
    print("Fran - Dataset Capture Edition")
    print("=" * 60)
    print("Down Arrow = Push To Talk")
    print("Left Arrow = Save Last Exchange")
    print("=" * 60)

    device_id = select_mic()

    load_all_models()

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    listener.start()

    threading.Thread(
        target=audio_loop,
        args=(device_id,),
        daemon=True
    ).start()

    threading.Thread(
        target=processor,
        daemon=True
    ).start()

    threading.Thread(
        target=twitch_thread,
        daemon=True
    ).start()

    print("\n[Fran] Running...\n")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        running = False
        print("\n[Fran] Shutdown")
        time.sleep(0.5)
        sys.exit(0)


if __name__ == "__main__":
    main()