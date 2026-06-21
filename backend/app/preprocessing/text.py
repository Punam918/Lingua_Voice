"""
Language-specific text normalization for TTS and transcript post-processing.
"""
import re
import unicodedata


# ─── Generic helpers ───────────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ─── Language-specific normalizers ────────────────────────────────────────────

def normalize_english(text: str) -> str:
    text = normalize_unicode(text)
    # Expand common contractions (minimal set)
    contractions = {
        r"\bdon't\b": "do not",
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bI'm\b": "I am",
        r"\bI've\b": "I have",
        r"\bI'll\b": "I will",
        r"\bI'd\b": "I would",
        r"\bit's\b": "it is",
        r"\bthey're\b": "they are",
        r"\bwe're\b": "we are",
        r"\byou're\b": "you are",
    }
    for pat, rep in contractions.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = normalize_whitespace(text)
    return text


def normalize_german(text: str) -> str:
    text = normalize_unicode(text)
    # Preserve umlauts; normalize typographic quotes to ASCII equivalents
    # „ = U+201E (LOW-9 QUOTATION MARK), " = U+201C (LEFT DOUBLE QUOTATION MARK)
    # ‚ = U+201A (SINGLE LOW-9 QUOTATION MARK), ' = U+2018 (LEFT SINGLE QUOTATION MARK)
    text = text.replace("\u201e", '"').replace("\u201c", '"')
    text = text.replace("\u201a", "'").replace("\u2018", "'")
    text = normalize_whitespace(text)
    return text


def normalize_spanish(text: str) -> str:
    text = normalize_unicode(text)
    # Preserve accented characters (NFC already handles decomposed forms)
    text = normalize_whitespace(text)
    return text


def normalize_french(text: str) -> str:
    text = normalize_unicode(text)
    # Normalize typographic quotes
    text = text.replace("«", '"').replace("»", '"')
    # Normalize apostrophes
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = normalize_whitespace(text)
    return text


def normalize_nepali(text: str) -> str:
    text = normalize_unicode(text)
    # Normalize danda (।) spacing
    text = re.sub(r"\s*।\s*", "। ", text)
    # Normalize double danda
    text = re.sub(r"\s*॥\s*", "॥ ", text)
    # Normalize digits: convert ASCII digits to Devanagari if desired
    # (kept as ASCII here since Whisper works better with Arabic digits)
    text = normalize_whitespace(text)
    return text


# ─── Router ────────────────────────────────────────────────────────────────────

_NORMALIZERS = {
    "english": normalize_english,
    "german":  normalize_german,
    "spanish": normalize_spanish,
    "french":  normalize_french,
    "nepali":  normalize_nepali,
}


def normalize(text: str, language: str) -> str:
    """Normalize text for a given language. Falls back to whitespace normalization."""
    fn = _NORMALIZERS.get(language, normalize_whitespace)
    return fn(text)
