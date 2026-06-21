"""
Text-to-Speech API router.

POST /api/synthesize
  - Accepts text + language + voice settings
  - Returns MP3 audio stream

GET /api/voices/{language}
  - Returns available voices for a language
"""
import logging

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse

from app.preprocessing.audio import make_tmp_path
from app.preprocessing.text import normalize
from app.utils.config import LANGUAGES, MAX_TEXT_CHARS

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/tts")           # canonical (per strategy spec)
@router.post("/api/synthesize")    # legacy alias (kept for compatibility)
async def synthesize(
    request: Request,
    text:     str = Form(...),
    language: str = Form("english"),
    gender:   str = Form("female"),
    rate:     str = Form("+0%"),
    pitch:    str = Form("+0Hz"),
):
    """
    Synthesize speech from text.

    Fields:
        text     – input text (max 2000 chars)
        language – language key
        gender   – 'male' or 'female'
        rate     – speed adjustment (e.g. '+10%', '-20%')
        pitch    – pitch adjustment (e.g. '+5Hz')
    """
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language}")

    text = text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text exceeds {MAX_TEXT_CHARS} character limit")

    # Normalize text for the language
    text = normalize(text, language)

    out_path = make_tmp_path(".mp3")
    try:
        tts = request.app.state.tts
        await tts.synthesize(
            text=text,
            language=language,
            gender=gender,
            rate=rate,
            pitch=pitch,
            out_path=out_path,
        )

        return FileResponse(
            str(out_path),
            media_type="audio/mpeg",
            filename=f"linguavoice_{language}.mp3",
            headers={
                "X-Language": language,
                "X-Gender":   gender,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("TTS error")
        if out_path.exists():
            out_path.unlink()
        raise HTTPException(500, str(exc)) from exc


@router.get("/api/voices/{language}")
async def list_voices(language: str, request: Request):
    """Return available Edge TTS voices for a language."""
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language}")
    tts = request.app.state.tts
    voices = await tts.list_voices(language)
    return {"language": language, "voices": voices}
