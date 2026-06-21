"""
Health endpoint.
"""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    state = request.app.state
    return {
        "status":       "ok",
        "stt_model":    getattr(state, "whisper_model_name", "unknown"),
        "tts_engine":   "edge-tts",
        "languages":    5,
    }
