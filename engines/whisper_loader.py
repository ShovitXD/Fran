from faster_whisper import WhisperModel

WHISPER_MODEL = None


def load_whisper(model_size="large-v3-turbo", device="cuda", compute_type="float16"):
    global WHISPER_MODEL
    print("[Fran] Loading Whisper...")
    WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("[Fran] Whisper Ready")
    return WHISPER_MODEL


def transcribe(audio, language="en", beam_size=5):
    segments, _ = WHISPER_MODEL.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=False,
    )
    return " ".join(s.text for s in segments).strip()