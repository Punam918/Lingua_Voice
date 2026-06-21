"""
Speech-to-Text API router.

POST /api/transcribe
  - Accepts an audio file upload (any format ffmpeg supports)
  - Returns transcript, detected language, segments, timing
"""
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.preprocessing.audio import make_tmp_path
from app.preprocessing.text import normalize
from app.utils.config import LANGUAGES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/stt")           # canonical (per strategy spec)
@router.post("/api/transcribe")    # legacy alias (kept for compatibility)
async def transcribe(
    request:     Request,
    audio:       UploadFile = File(...),
    language:    str        = Form("english"),
    auto_detect: bool       = Form(False),
):
    """
    Transcribe uploaded audio.

    Fields:
        audio       – audio file (webm, wav, mp3, ogg, m4a, …)
        language    – selected language key (ignored if auto_detect=true)
        auto_detect – if true, Whisper detects the language automatically
    """
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language}")

    suffix   = Path(audio.filename).suffix if audio.filename else ".webm"
    tmp_path = make_tmp_path(suffix)

    try:
        tmp_path.write_bytes(await audio.read())

        whisper   = request.app.state.whisper
        lang_code = None if auto_detect else LANGUAGES[language]["code"]

        result = whisper.transcribe(str(tmp_path), language=lang_code)

        # Post-process transcript
        cleaned_text = normalize(result["text"], language)

        return {
            "transcript":        cleaned_text,          # canonical name (strategy spec)
            "text":              cleaned_text,           # alias for frontend compat
            "language":          language,
            "detected_language": result.get("detected_language"),
            "confidence":        result.get("confidence", 0.0),
            "duration_sec":      result["elapsed_sec"],
            "elapsed_sec":       result["elapsed_sec"],  # legacy alias
            "segments":          result["segments"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("STT error")
        raise HTTPException(500, str(exc)) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
