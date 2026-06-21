"""
Centralized language + model configuration for LinguaVoice.
"""
import os

# ─── Language Registry ─────────────────────────────────────────────────────────

LANGUAGES: dict[str, dict] = {
    "english": {
        "code": "en",
        "label": "English",
        "flag": "🇬🇧",
        "placeholder": "Hello, how are you today?",
        "tts_voice_male":   "en-US-GuyNeural",
        "tts_voice_female": "en-US-JennyNeural",
        "xtts_lang": "en",
    },
    "german": {
        "code": "de",
        "label": "Deutsch",
        "flag": "🇩🇪",
        "placeholder": "Guten Tag, wie geht es Ihnen?",
        "tts_voice_male":   "de-DE-ConradNeural",
        "tts_voice_female": "de-DE-KatjaNeural",
        "xtts_lang": "de",
    },
    "spanish": {
        "code": "es",
        "label": "Español",
        "flag": "🇪🇸",
        "placeholder": "Hola, ¿cómo estás hoy?",
        "tts_voice_male":   "es-ES-AlvaroNeural",
        "tts_voice_female": "es-ES-ElviraNeural",
        "xtts_lang": "es",
    },
    "french": {
        "code": "fr",
        "label": "Français",
        "flag": "🇫🇷",
        "placeholder": "Bonjour, comment allez-vous?",
        "tts_voice_male":   "fr-FR-HenriNeural",
        "tts_voice_female": "fr-FR-DeniseNeural",
        "xtts_lang": "fr",
    },
    "nepali": {
        "code": "ne",
        "label": "नेपाली",
        "flag": "🇳🇵",
        "placeholder": "नमस्ते, तपाईलाई कस्तो छ?",
        "tts_voice_male":   "ne-NP-SagarNeural",
        "tts_voice_female": "ne-NP-HemkalaNeural",
        # Nepali is NOT in XTTS-v2 official support list → use Edge TTS / MMS fallback
        "xtts_lang": None,
    },
}

# Languages supported by XTTS-v2 for fine-tuning
XTTS_SUPPORTED = {"english", "german", "spanish", "french"}

# Languages routed to MMS-TTS
MMS_SUPPORTED = {"nepali"}

# ─── Model paths (override via env) ────────────────────────────────────────────

WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL", "medium")
FINETUNED_MODEL_DIR = os.getenv("FINETUNED_MODEL_DIR", "")   # if set, load from here
XTTS_MODEL_DIR      = os.getenv("XTTS_MODEL_DIR", "")        # custom XTTS checkpoint
MMS_MODEL_DIR       = os.getenv("MMS_MODEL_DIR", "")         # custom Nepali MMS checkpoint

# ─── Audio constants ───────────────────────────────────────────────────────────

TARGET_SAMPLE_RATE = 16_000   # Whisper input
TTS_SAMPLE_RATE    = 24_000   # XTTS output
MAX_RECORD_SECS    = 180
MAX_TEXT_CHARS     = 2000
