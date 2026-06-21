"""
XTTS-v2 TTS Service (Coqui AI).

Provides high-quality neural TTS for English, German, Spanish, French.
Nepali is NOT in XTTS-v2's official supported language list → use EdgeTTSService.

Requires: pip install TTS  (Coqui TTS)
GPU strongly recommended; CPU is very slow.

Model: tts_models/multilingual/multi-dataset/xtts_v2
"""
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# XTTS-v2 official language codes
XTTS_LANG_MAP = {
    "english": "en",
    "german":  "de",
    "spanish": "es",
    "french":  "fr",
}

# XTTS-v2 does not support Nepali
XTTS_UNSUPPORTED = {"nepali"}


class XTTSService:
    """
    Wraps Coqui XTTS-v2 for high-quality multilingual TTS.

    Supports voice cloning via a reference audio sample.
    For languages not supported by XTTS, fall back to EdgeTTSService.
    """

    def __init__(self, model_dir: str = "", use_gpu: bool = True):
        """
        Args:
            model_dir: Path to a fine-tuned XTTS checkpoint directory.
                       If empty, loads the pretrained XTTS-v2 from Coqui.
            use_gpu:   Use CUDA if available.
        """
        from TTS.api import TTS
        import torch

        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        t0 = time.time()
        if model_dir and Path(model_dir).exists():
            logger.info(f"[XTTSService] Loading fine-tuned checkpoint: {model_dir}")
            self.tts = TTS(model_path=model_dir, config_path=str(Path(model_dir) / "config.json"))
        else:
            logger.info("[XTTSService] Loading pretrained XTTS-v2…")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        self.tts.to(device)
        logger.info(f"[XTTSService] Ready on {device} in {time.time()-t0:.1f}s")

    def synthesize(
        self,
        text: str,
        language: str,
        out_path: Path,
        speaker_wav: Optional[str] = None,
        speed: float = 1.0,
    ) -> Path:
        """
        Synthesize speech with XTTS-v2.

        Args:
            text:        Input text.
            language:    Language key (e.g. 'english').
            out_path:    Output WAV path.
            speaker_wav: Reference audio for voice cloning (optional).
                         If None, uses XTTS built-in speaker.
            speed:       Speaking speed multiplier.

        Returns Path to the generated WAV file.
        """
        if language in XTTS_UNSUPPORTED:
            raise ValueError(
                f"XTTS-v2 does not support '{language}'. "
                "Use EdgeTTSService or MMSTTSService for this language."
            )

        xtts_lang = XTTS_LANG_MAP.get(language)
        if not xtts_lang:
            raise ValueError(f"Unknown language for XTTS: {language}")

        logger.info(f"[XTTSService] Synthesizing text ({len(text)} chars) → {out_path.name}")

        tts_kwargs = dict(
            text=text,
            language=xtts_lang,
            file_path=str(out_path),
            speed=speed,
        )
        if speaker_wav:
            tts_kwargs["speaker_wav"] = speaker_wav

        self.tts.tts_to_file(**tts_kwargs)
        return out_path

    @property
    def supported_languages(self) -> list[str]:
        return list(XTTS_LANG_MAP.keys())
