"""
Languages endpoint – returns metadata for all supported languages.
"""
from fastapi import APIRouter

from app.utils.config import LANGUAGES

router = APIRouter()


@router.get("/api/languages")
def get_languages():
    return {
        key: {
            "label":       cfg["label"],
            "flag":        cfg["flag"],
            "placeholder": cfg["placeholder"],
            "code":        cfg["code"],
        }
        for key, cfg in LANGUAGES.items()
    }
