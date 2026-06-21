"""
Prepare your own audio data for Whisper fine-tuning.

Expected input folder structure:
    data/
        english/
            audio/   *.wav / *.mp3
            labels.csv   (columns: filename, transcript)
        german/
            audio/
            labels.csv
        ...

Run:
    python prepare_custom_data.py --language nepali --data_dir ./data --out_dir ./processed
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 16_000  # Whisper expects 16 kHz mono


def process_audio(src_path: Path, dst_path: Path):
    """Resample + convert to mono 16 kHz WAV."""
    audio, sr = librosa.load(str(src_path), sr=TARGET_SR, mono=True)
    sf.write(str(dst_path), audio, TARGET_SR)


def prepare(language: str, data_dir: Path, out_dir: Path):
    lang_dir = data_dir / language
    audio_dir = lang_dir / "audio"
    labels_file = lang_dir / "labels.csv"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_file}")

    out_lang = out_dir / language
    out_audio = out_lang / "audio"
    out_audio.mkdir(parents=True, exist_ok=True)

    manifest = []

    with open(labels_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} samples for {language}…")

    for row in rows:
        filename   = row["filename"].strip()
        transcript = row["transcript"].strip()

        src = audio_dir / filename
        if not src.exists():
            print(f"  [SKIP] {src} not found")
            continue

        dst = out_audio / (src.stem + ".wav")
        process_audio(src, dst)

        manifest.append({
            "audio_filepath": str(dst.resolve()),
            "text": transcript,
            "duration": librosa.get_duration(filename=str(dst)),
        })

    manifest_path = out_lang / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(manifest)} samples → {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language",  required=True,
                        choices=["english", "german", "spanish", "french", "nepali"])
    parser.add_argument("--data_dir",  type=Path, default=Path("./data"))
    parser.add_argument("--out_dir",   type=Path, default=Path("./processed"))
    args = parser.parse_args()

    prepare(args.language, args.data_dir, args.out_dir)
