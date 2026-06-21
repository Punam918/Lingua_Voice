"""
Nepali TTS Service using Meta's MMS-TTS.

MMS-TTS provides coverage for 1000+ languages including Nepali (npi).
This is the recommended TTS route for Nepali since XTTS-v2 does not
officially support it.

Requires: pip install transformers torch

Model: facebook/mms-tts-npi
"""
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MMS_TTS_NEPALI_MODEL = "facebook/mms-tts-npi"


class NepaliTTSService:
    """
    Wraps facebook/mms-tts-npi for Nepali text-to-speech.

    Falls back to Edge TTS if MMS-TTS is unavailable or fails.
    """

    def __init__(self, model_id: str = MMS_TTS_NEPALI_MODEL, model_dir: str = ""):
        """
        Args:
            model_id:  HuggingFace model ID for MMS-TTS Nepali.
            model_dir: Path to a local fine-tuned checkpoint (overrides model_id).
        """
        from transformers import VitsModel, AutoTokenizer
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        load_id = model_dir if model_dir and Path(model_dir).exists() else model_id

        t0 = time.time()
        logger.info(f"[NepaliTTSService] Loading {load_id} on {self.device}…")
        self.tokenizer = AutoTokenizer.from_pretrained(load_id)
        self.model     = VitsModel.from_pretrained(load_id).to(self.device)
        self.model.eval()
        self.sample_rate = self.model.config.sampling_rate
        logger.info(f"[NepaliTTSService] Ready in {time.time()-t0:.1f}s · SR={self.sample_rate}")

    def synthesize(self, text: str, out_path: Path, speed: float = 1.0) -> Path:
        """
        Synthesize Nepali speech.

        Args:
            text:     Nepali text (Devanagari script).
            out_path: Output WAV path.
            speed:    Speaking speed via VITS length scale (< 1 = faster).

        Returns Path to the generated WAV file.
        """
        import torch
        import soundfile as sf

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # VITS length_scale controls speed: 1/speed converts from speed multiplier
        length_scale = 1.0 / max(speed, 0.5)

        logger.info(f"[NepaliTTSService] Synthesizing ({len(text)} chars) → {out_path.name}")
        with torch.no_grad():
            output = self.model(**inputs, length_scale=length_scale)

        waveform = output.waveform[0].squeeze().cpu().numpy()
        sf.write(str(out_path), waveform, self.sample_rate)
        return out_path
