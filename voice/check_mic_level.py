import sounddevice as sd
import numpy as np

devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        print(f"[{i}] {dev['name']}")

device_id = int(input("\nPick device number: "))

print(f"\nUsing: {sd.query_devices(device_id)['name']}")
print("Speak now... Ctrl+C to stop.\n")

try:
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32", device=device_id) as stream:
        while True:
            chunk, _ = stream.read(1600)
            if chunk.size == 0:
                continue
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            bar = "█" * int(rms * 1000)
            print(f"RMS: {rms:.6f}  {bar}          ", end="\r")
except KeyboardInterrupt:
    print("\nDone.")
except Exception as e:
    print(f"\nError on this device: {e}")
    print("Try a different device number.")