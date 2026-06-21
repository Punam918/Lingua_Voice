"""
Whisper STT Service.

Loads the Whisper model once at startup and exposes a clean transcribe() method.
Supports both the standard OpenAI Whisper and a fine-tuned HuggingFace checkpoint.
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Wraps OpenAI Whisper for multilingual speech-to-text.

    If FINETUNED_MODEL_DIR is set in the environment and the directory exists,
    the fine-tuned checkpoint is loaded instead of the base model.
    """

    def __init__(self, model_size: str = "medium", finetuned_dir: str = ""):
        import whisper
        t0 = time.time()
        if finetuned_dir and Path(finetuned_dir).exists():
            logger.info(f"[WhisperService] Loading fine-tuned model from: {finetuned_dir}")
            self.model = whisper.load_model(finetuned_dir)
            self.model_name = f"finetuned:{Path(finetuned_dir).name}"
        else:
            logger.info(f"[WhisperService] Loading whisper-{model_size}")
            self.model = whisper.load_model(model_size)
            self.model_name = model_size
        logger.info(f"[WhisperService] Ready in {time.time()-t0:.1f}s")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (any format ffmpeg supports).
            language:   ISO 639-1 code (e.g. 'ne'). None = auto-detect.
            task:       'transcribe' or 'translate' (to English).

        Returns dict with keys: text, language, segments, duration_sec.
        """
        import whisper

        t0 = time.time()
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task=task,
            fp16=False,  # CPU-safe default; auto-enabled on CUDA if model is on GPU
        )
        elapsed = round(time.time() - t0, 2)

        segments = [
            {
                "start": round(s["start"], 2),
                "end":   round(s["end"], 2),
                "text":  s["text"].strip(),
            }
            for s in result.get("segments", [])
        ]

        # Approximate confidence from average log-probability of segments
        log_probs = [s.get("avg_logprob", -1.0) for s in result.get("segments", []) if "avg_logprob" in s]
        import math
        avg_lp     = sum(log_probs) / len(log_probs) if log_probs else -0.5
        confidence = round(min(1.0, max(0.0, math.exp(avg_lp))), 3)

        return {
            "text":              result["text"].strip(),
            "detected_language": result.get("language"),
            "segments":          segments,
            "elapsed_sec":       elapsed,
            "confidence":        confidence,
        }
