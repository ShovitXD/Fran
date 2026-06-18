from kokoro import KPipeline
import sounddevice as sd

pipeline = KPipeline(lang_code="a")

generator = pipeline(
    "Hello. My name is Fran.",
    voice="af_heart"
)

for _, _, audio in generator:
    sd.play(audio, 24000)
    sd.wait()

print("Finished")