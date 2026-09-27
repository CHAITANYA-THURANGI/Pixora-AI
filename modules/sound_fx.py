"""
Sci-Fi Cybernetic Audio Synthesizer — v2.5.

Plays synthesized real-time audio cues for target locking, sarcasm alerts,
meme shutter capture, and GIF recording in non-blocking threads.
"""

import sys
import threading
import time

_sound_enabled = True


def _beep_async(freq, duration_ms):
    """Plays a synthesized frequency tone asynchronously on Windows."""
    if not _sound_enabled:
        return
    if sys.platform == "win32":
        def _play():
            try:
                import winsound
                winsound.Beep(int(freq), int(duration_ms))
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()


def play_target_lock():
    """Cyberpunk target lock chirp."""
    if not _sound_enabled:
        return
    def _seq():
        try:
            import winsound
            winsound.Beep(980, 40)
            time.sleep(0.02)
            winsound.Beep(1480, 50)
        except Exception:
            pass
    threading.Thread(target=_seq, daemon=True).start()


def play_sarcasm_alert():
    """High sarcasm probability alert."""
    if not _sound_enabled:
        return
    def _seq():
        try:
            import winsound
            winsound.Beep(1200, 60)
            time.sleep(0.03)
            winsound.Beep(1800, 80)
        except Exception:
            pass
    threading.Thread(target=_seq, daemon=True).start()


def play_shutter():
    """Meme capture shutter sound."""
    _beep_async(1760, 45)


def play_record_start():
    """GIF recording started."""
    _beep_async(880, 70)


def play_record_end():
    """GIF recording completed."""
    _beep_async(1320, 80)


def toggle_sound_fx() -> bool:
    """Toggles sound effects ON/OFF."""
    global _sound_enabled
    _sound_enabled = not _sound_enabled
    return _sound_enabled


def is_sound_fx_enabled() -> bool:
    return _sound_enabled
