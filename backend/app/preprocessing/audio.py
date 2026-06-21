"""
Audio preprocessing utilities.

All functions are stateless and work on file paths or numpy arrays.
"""
import tempfile
import uuid
from pathlib import Path

import numpy as np

TMP_DIR = Path(tempfile.gettempdir()) / "linguavoice_audio"
TMP_DIR.mkdir(exist_ok=True)


def make_tmp_path(suffix: str = ".wav") -> Path:
    return TMP_DIR / f"{uuid.uuid4()}{suffix}"


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio array to target sample rate using librosa."""
    import librosa
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo/multi-channel to mono."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0)


def normalize_volume(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Peak-normalize audio to a target dBFS level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-8:
        return audio
    target_rms = 10 ** (target_dbfs / 20)
    return audio * (target_rms / rms)


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 40) -> np.ndarray:
    """Trim leading/trailing silence."""
    import librosa
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def load_audio(path: str | Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """
    Load any audio file, convert to mono 16 kHz.
    Returns (array, sample_rate).
    """
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio, sr


def save_wav(audio: np.ndarray, path: str | Path, sr: int = 16_000) -> None:
    """Save numpy array as WAV."""
    import soundfile as sf
    sf.write(str(path), audio, sr)


def convert_to_wav(src: str | Path, dst: str | Path, target_sr: int = 16_000) -> None:
    """Convert any audio format to mono WAV at target_sr."""
    audio, _ = load_audio(src, target_sr=target_sr)
    save_wav(audio, dst, sr=target_sr)


def get_duration(path: str | Path) -> float:
    """Return audio duration in seconds."""
    import librosa
    return float(librosa.get_duration(path=str(path)))
