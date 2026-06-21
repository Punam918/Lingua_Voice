# ─── Compatibility shim ───────────────────────────────────────────────────────
# This file is kept for backward compatibility with existing Docker/run scripts.
# The actual application now lives in app/main.py (modular architecture).
# New imports, services, and API routes are defined there.
#
# To run the modular version directly:
#   uvicorn app.main:app --host 0.0.0.0 --port 8000
# ─────────────────────────────────────────────────────────────────────────────

import os
import asyncio
import tempfile
import uuid
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import whisper
import edge_tts

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="VoiceBridge API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Language Config ──────────────────────────────────────────────────────────

LANGUAGES = {
    "english": {
        "whisper_code": "en",
        "tts_voice_male":   "en-US-GuyNeural",
        "tts_voice_female": "en-US-JennyNeural",
        "label": "English",
        "flag": "🇬🇧",
        "placeholder": "Hello, how are you today?",
    },
    "german": {
        "whisper_code": "de",
        "tts_voice_male":   "de-DE-ConradNeural",
        "tts_voice_female": "de-DE-KatjaNeural",
        "label": "Deutsch",
        "flag": "🇩🇪",
        "placeholder": "Guten Tag, wie geht es Ihnen?",
    },
    "spanish": {
        "whisper_code": "es",
        "tts_voice_male":   "es-ES-AlvaroNeural",
        "tts_voice_female": "es-ES-ElviraNeural",
        "label": "Español",
        "flag": "🇪🇸",
        "placeholder": "Hola, ¿cómo estás hoy?",
    },
    "french": {
        "whisper_code": "fr",
        "tts_voice_male":   "fr-FR-HenriNeural",
        "tts_voice_female": "fr-FR-DeniseNeural",
        "label": "Français",
        "flag": "🇫🇷",
        "placeholder": "Bonjour, comment allez-vous?",
    },
    "nepali": {
        "whisper_code": "ne",
        "tts_voice_male":   "ne-NP-SagarNeural",
        "tts_voice_female": "ne-NP-HemkalaNeural",
        "label": "नेपाली",
        "flag": "🇳🇵",
        "placeholder": "नमस्ते, तपाईलाई कस्तो छ?",
    },
}

# ─── Model Loading ────────────────────────────────────────────────────────────

_startup_time = time.time()
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")
print(f"[VoiceBridge] Loading Whisper '{WHISPER_MODEL_SIZE}'…")
stt_model = whisper.load_model(WHISPER_MODEL_SIZE)
print(f"[VoiceBridge] Model ready in {time.time()-_startup_time:.1f}s")

TMP_DIR = Path(tempfile.gettempdir()) / "voicebridge_audio"
TMP_DIR.mkdir(exist_ok=True)

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": WHISPER_MODEL_SIZE}

# ─── Languages ───────────────────────────────────────────────────────────────

@app.get("/api/languages")
def get_languages():
    return {
        k: {kk: vv for kk, vv in v.items() if kk != "whisper_code"}
        for k, v in LANGUAGES.items()
    }

# ─── STT ─────────────────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("english"),
    auto_detect: bool = Form(False),
):
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language}")

    suffix = Path(audio.filename).suffix if audio.filename else ".webm"
    tmp_path = TMP_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        tmp_path.write_bytes(await audio.read())

        lang_code = None if auto_detect else LANGUAGES[language]["whisper_code"]
        t0 = time.time()
        result = stt_model.transcribe(str(tmp_path), language=lang_code, fp16=False)
        elapsed = round(time.time() - t0, 2)

        return {
            "text": result["text"].strip(),
            "language": language,
            "detected_language": result.get("language"),
            "elapsed_sec": elapsed,
            "segments": [
                {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

# ─── TTS ─────────────────────────────────────────────────────────────────────

@app.post("/api/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("english"),
    gender: str = Form("female"),
    rate: str = Form("+0%"),
    pitch: str = Form("+0Hz"),
):
    if language not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language}")
    if not text.strip():
        raise HTTPException(400, "Text cannot be empty")

    voice_key = "tts_voice_male" if gender == "male" else "tts_voice_female"
    voice = LANGUAGES[language][voice_key]

    out_path = TMP_DIR / f"{uuid.uuid4()}.mp3"
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(str(out_path))

        return FileResponse(
            str(out_path),
            media_type="audio/mpeg",
            filename="voicebridge_speech.mp3",
            headers={"X-Voice": voice, "X-Language": language},
        )
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        raise HTTPException(500, str(e))

# ─── Voices ──────────────────────────────────────────────────────────────────

@app.get("/api/voices/{language}")
async def list_voices(language: str):
    if language not in LANGUAGES:
        raise HTTPException(400, "Unsupported language")
    cfg = LANGUAGES[language]
    all_voices = await edge_tts.list_voices()
    lang_prefix = cfg["whisper_code"].lower()
    return {
        "language": language,
        "voices": [
            {"name": v["ShortName"], "gender": v["Gender"], "locale": v["Locale"]}
            for v in all_voices
            if v["Locale"].lower().startswith(lang_prefix)
        ],
    }

# ─── Serve Frontend ───────────────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(frontend_dir / "index.html"))

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
