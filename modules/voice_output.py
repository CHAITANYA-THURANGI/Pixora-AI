"""
Empathic Offline AI Voice Persona (TTS) — v2.5.

Speaks sarcastic verdicts, generated meme captions, and system alerts using
offline speech synthesis. Runs in a dedicated non-blocking thread so live video
FPS is never affected.
"""

import queue
import threading
import time

_speech_queue = queue.Queue()
_tts_thread = None
_tts_running = True
_speech_enabled = True


def _tts_worker():
    """Background worker thread that consumes speech requests sequentially."""
    engine = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)  # slightly brisk, natural conversational speed
        engine.setProperty("volume", 0.9)
    except Exception as e:
        engine = None

    while _tts_running:
        try:
            text = _speech_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if text and _speech_enabled:
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            else:
                # Windows SAPI fallback
                try:
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(text)
                except Exception:
                    pass
        _speech_queue.task_done()


def init_voice():
    """Initializes the background speech synthesizer thread."""
    global _tts_thread
    if _tts_thread is None or not _tts_thread.is_alive():
        _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        _tts_thread.start()


def speak_async(text: str):
    """Queues a sentence to be spoken asynchronously by the AI persona."""
    if not text or not _speech_enabled:
        return
    # Clean text
    clean_text = text.replace('"', '').replace("'", "").strip()
    if clean_text:
        init_voice()
        _speech_queue.put(clean_text)


def toggle_speech() -> bool:
    """Toggles voice synthesis ON/OFF."""
    global _speech_enabled
    _speech_enabled = not _speech_enabled
    return _speech_enabled


def is_speech_enabled() -> bool:
    return _speech_enabled
