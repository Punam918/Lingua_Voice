"""
Edge TTS Service (Microsoft Neural Voices).

Provides high-quality neural TTS for all 5 languages via edge-tts.
This is the primary TTS engine – fast, free, no GPU required.
"""
import asyncio
import logging
from pathlib import Path
from typing import Optional

import edge_tts

from app.utils.config import LANGUAGES
from app.preprocessing.audio import make_tmp_path

logger = logging.getLogger(__name__)


class EdgeTTSService:
    """
    Wraps edge-tts for multilingual text-to-speech.

    Microsoft's Neural TTS covers all 5 target languages including Nepali
    (ne-NP-SagarNeural / ne-NP-HemkalaNeural).
    """

    def __init__(self):
        logger.info("[EdgeTTSService] Ready (no model load required)")

    async def synthesize(
        self,
        text: str,
        language: str,
        gender: str = "female",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        out_path: Optional[Path] = None,
    ) -> Path:
        """
        Synthesize speech and save to a file.

        Args:
            text:     Input text.
            language: Language key (e.g. 'english', 'nepali').
            gender:   'male' or 'female'.
            rate:     Speed adjustment (e.g. '+10%', '-20%').
            pitch:    Pitch adjustment (e.g. '+5Hz', '-10Hz').
            out_path: Output path. Auto-generated if None.

        Returns Path to the generated MP3 file.
        """
        cfg = LANGUAGES.get(language)
        if not cfg:
            raise ValueError(f"Unsupported language: {language}")

        voice_key = "tts_voice_male" if gender == "male" else "tts_voice_female"
        voice = cfg[voice_key]

        if out_path is None:
            out_path = make_tmp_path(".mp3")

        logger.info(f"[EdgeTTSService] Synthesizing {language}/{gender} → {out_path.name}")
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(out_path))
        return out_path

    async def list_voices(self, language: str) -> list[dict]:
        """List all available Edge TTS voices for a language."""
        cfg = LANGUAGES.get(language)
        if not cfg:
            raise ValueError(f"Unsupported language: {language}")
        lang_prefix = cfg["code"].lower()
        all_voices = await edge_tts.list_voices()
        return [
            {
                "name":   v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
            }
            for v in all_voices
            if v["Locale"].lower().startswith(lang_prefix)
        ]
