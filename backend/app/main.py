"""
LinguaVoice – Multilingual Speech AI Platform
FastAPI application entry point (modular version).

Architecture:
  app/api/health.py    → GET  /health
  app/api/languages.py → GET  /api/languages
  app/api/stt.py       → POST /api/transcribe
  app/api/tts.py       → POST /api/synthesize, GET /api/voices/{lang}

Services are loaded once at startup and stored on app.state:
  app.state.whisper   → WhisperService
  app.state.tts       → EdgeTTSService
"""
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import health, languages, stt, tts
from app.services.whisper_service import WhisperService
from app.services.edge_tts_service import EdgeTTSService
from app.utils.config import WHISPER_MODEL_SIZE, FINETUNED_MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LinguaVoice API",
    version="2.0.0",
    description="Multilingual Speech-to-Text and Text-to-Speech for EN/DE/ES/FR/NE",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(languages.router)
app.include_router(stt.router)
app.include_router(tts.router)

# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("=== LinguaVoice startup ===")

    app.state.whisper = WhisperService(
        model_size=WHISPER_MODEL_SIZE,
        finetuned_dir=FINETUNED_MODEL_DIR,
    )
    app.state.whisper_model_name = app.state.whisper.model_name

    app.state.tts = EdgeTTSService()

    logger.info("=== All services ready ===")


# ─── Serve Frontend ───────────────────────────────────────────────────────────

_frontend = Path(__file__).parent.parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(_frontend / "index.html"))


# ─── Dev entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
