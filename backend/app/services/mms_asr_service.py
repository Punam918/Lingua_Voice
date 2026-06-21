"""
MMS ASR Service (Meta's Massively Multilingual Speech).

Used as an alternative/fallback ASR for Nepali and other low-resource languages.
Requires: pip install transformers torch torchaudio

Model: facebook/mms-300m (or facebook/mms-1b-all for broader language coverage)
"""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# MMS language codes (ISO 639-3 for some; MMS uses its own codes for some)
MMS_LANG_MAP = {
    "english": "eng",
    "german":  "deu",
    "spanish": "spa",
    "french":  "fra",
    "nepali":  "npi",
}


class MMSASRService:
    """
    Wraps facebook/mms-300m for multilingual ASR.

    Particularly useful for Nepali where Whisper may underperform.
    Can be used as a comparison baseline or primary engine per language.
    """

    def __init__(self, model_id: str = "facebook/mms-300m"):
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        logger.info(f"[MMSASRService] Loading {model_id} on {self.device}…")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()
        logger.info(f"[MMSASRService] Ready in {time.time()-t0:.1f}s")

    def transcribe(
        self,
        audio_path: str,
        language: str = "nepali",
        target_sr: int = 16_000,
    ) -> dict:
        """
        Transcribe audio using MMS ASR.

        Args:
            audio_path: Path to audio file.
            language:   Language key (e.g. 'nepali').
            target_sr:  Sample rate (MMS expects 16 kHz).

        Returns dict with keys: text, language, elapsed_sec.
        """
        import torch
        import librosa

        mms_lang = MMS_LANG_MAP.get(language, "npi")

        # Set language token on the processor
        self.processor.tokenizer.set_target_lang(mms_lang)
        self.model.load_adapter(mms_lang)

        audio, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)

        t0 = time.time()
        inputs = self.processor(audio, sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        text = self.processor.decode(ids)
        elapsed = round(time.time() - t0, 2)

        return {
            "text":        text.strip(),
            "language":    language,
            "elapsed_sec": elapsed,
        }
