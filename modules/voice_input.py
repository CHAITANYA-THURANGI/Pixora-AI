"""
Voice input for sarcasm detection -- records a few seconds from your
microphone and transcribes it locally with OpenAI's Whisper model, so the
SAME text-based sarcasm pipeline (modules/sarcasm_text.py) can score
spoken sarcasm, not just typed sarcasm. Fully local and free: no cloud
speech API, no internet needed after the one-time model download.

Requires requirements-advanced.txt (openai-whisper, sounddevice). If
either isn't installed, or no microphone is found, this fails gracefully
and main.py falls back to typed input -- voice is a bonus, not a
dependency the rest of the app relies on.
"""

import numpy as np

import config

_whisper_model = None
_import_error = None

try:
    import sounddevice as sd
    import whisper
except Exception as e:  # pragma: no cover - exact exception type varies by platform
    sd = None
    whisper = None
    _import_error = e


def voice_input_available() -> bool:
    return _import_error is None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"[voice_input] Loading Whisper model '{config.WHISPER_MODEL}' "
              f"(one-time download on first run)...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL)
    return _whisper_model


def record_audio(duration=None, sample_rate=16000):
    """
    Records mono float32 audio directly at Whisper's native 16kHz sample
    rate (avoids needing ffmpeg-based resampling/decoding -- this array
    goes straight into whisper.transcribe()).
    """
    duration = duration or config.VOICE_RECORD_SECONDS
    print(f"[voice_input] Recording for {duration}s -- speak now...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[voice_input] Done recording.")
    return audio.flatten()


def transcribe(audio_array: np.ndarray) -> str:
    """Runs Whisper on an already-captured 16kHz mono float32 array."""
    model = _get_model()
    import torch
    result = model.transcribe(audio_array, fp16=torch.cuda.is_available(), language="en")
    return result["text"].strip()


def record_and_transcribe(duration=None) -> str:
    """
    One-call convenience: record from mic, transcribe, return text.
    Returns "" (not an exception) on any failure, so callers can just
    check `if text:` rather than wrapping this in their own try/except.
    """
    if not voice_input_available():
        print(f"[voice_input] Not available ({_import_error}). "
              f"Install requirements-advanced.txt for voice input, or use typed input instead.")
        return ""
    try:
        audio = record_audio(duration)
        text = transcribe(audio)
        if text:
            print(f'[voice_input] Heard: "{text}"')
        else:
            print("[voice_input] Didn't catch anything -- silence or unclear audio.")
        return text
    except Exception as e:
        print(f"[voice_input] Failed ({e}). Falling back to typed input is fine -- "
              f"check your microphone is connected and not in use by another app.")
        return ""
