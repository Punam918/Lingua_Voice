# LinguaVoice

LinguaVoice is a multilingual speech AI platform built as a browser app plus a FastAPI backend. It supports:

- speech to text from the microphone or uploaded audio
- text to speech with language, voice, speed, and pitch controls
- five target languages: English, German, Spanish, French, Nepali
- model fine-tuning and evaluation pipelines for STT and TTS
- local and Docker-based deployment

This repository is not a chat assistant or a general-purpose LLM app. It does not currently ship a conversational open-source LLM in the runtime path. Instead, it uses specialized speech models:

- Whisper for speech recognition
- Edge TTS for runtime speech synthesis
- XTTS-v2 for multilingual TTS training and future inference paths
- MMS-based models for Nepali ASR/TTS support

That is the right design for this project because the product is speech-first, not chat-first.

## What The App Does

### User-facing features

- Record speech in the browser and get a transcript back
- Upload an audio file and transcribe it
- Auto-detect the spoken language
- Switch between five supported languages
- Paste or type text and synthesize speech
- Choose male or female voices for TTS
- Adjust speaking rate and pitch
- Copy transcripts and send them directly into the TTS box
- Play and download generated audio
- View a session history of recent STT and TTS actions

### Model and data features

- Whisper inference at runtime
- Optional fine-tuned Whisper checkpoints
- Common Voice data download and preprocessing
- Whisper fine-tuning scripts
- XTTS fine-tuning scripts for EN/DE/ES/FR
- Nepali MMS-TTS fine-tuning scripts
- Dataset validation and synthetic bootstrap tools

## Open-Source Model Stack

### Runtime stack

| Function | Current runtime model |
|---|---|
| STT | OpenAI Whisper |
| TTS | Microsoft Edge TTS voices |
| Language normalization | Custom code |

### Training stack

| Function | Open-source model |
|---|---|
| STT fine-tuning | Whisper |
| Multilingual TTS fine-tuning | Coqui XTTS-v2 |
| Nepali ASR / alternative STT | Meta MMS ASR |
| Nepali TTS | MMS-TTS |

### LLM status

- No chat LLM is wired into the current app.
- No Ollama, vLLM, or local GPT-style server is required to run the product.
- The repo focuses on speech pipelines, not text generation.
- If you want an LLM layer later, the right place would be a separate summarization, translation, or assistant feature, not the core STT/TTS route.

## Why This Design

The repo uses specialized models instead of a single large LLM because:

- STT and TTS each have dedicated high-quality model families
- the app needs low-latency speech UX, not dialogue generation
- Nepali support is better handled by a model path built for low-resource speech
- this keeps runtime simpler and cheaper to operate

## Project Layout

```text
LinguaVoice/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── preprocessing/
│   │   ├── services/
│   │   ├── utils/
│   │   └── main.py
│   ├── finetune/
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── download_common_voice.py
│   └── collect_tts_data.py
├── frontend/
│   ├── index.html
│   ├── js/app.js
│   └── css/style.css
├── training/
│   ├── stt/
│   └── tts/
├── docker-compose.yml
├── docker-start.sh
├── run.sh
├── setup.sh
├── strategy.txt
└── README.md
```

## Runtime Architecture

### Frontend

The frontend is a single-page app. It:

- captures audio through `MediaRecorder`
- uploads files for transcription
- updates language accents and placeholders dynamically
- renders transcript segments and metadata
- controls TTS playback and download
- keeps a short in-memory history for the current page session

The frontend logic lives in [frontend/js/app.js](./frontend/js/app.js).

### Backend

The active backend entrypoint is [backend/app/main.py](./backend/app/main.py). It:

- creates the FastAPI app
- loads Whisper on startup
- loads the TTS service on startup
- mounts the frontend static files when present
- exposes health, language, STT, and TTS routes

The current backend routers are:

- [backend/app/api/health.py](./backend/app/api/health.py)
- [backend/app/api/languages.py](./backend/app/api/languages.py)
- [backend/app/api/stt.py](./backend/app/api/stt.py)
- [backend/app/api/tts.py](./backend/app/api/tts.py)

### Legacy compatibility path

There is also a legacy file:

- [backend/main.py](./backend/main.py)

It exists for backward compatibility with older launch scripts. The modular app in `backend/app/` is the current source of truth.

## End-to-End Flow

### Speech to Text

1. The user records audio or uploads a file.
2. The browser sends a multipart form to `POST /api/stt` or `POST /api/transcribe`.
3. The backend writes the audio to a temporary file.
4. Whisper transcribes the clip.
5. The text is normalized for the selected language.
6. The response includes transcript text, detected language, confidence, duration, and segments.
7. The frontend renders the transcript and offers copy/send-to-TTS actions.

### Text to Speech

1. The user types or pastes text.
2. The browser sends a multipart form to `POST /api/tts` or `POST /api/synthesize`.
3. The backend validates and normalizes the text.
4. The TTS engine generates an MP3.
5. The frontend plays the audio, shows the player, and enables download.

## UI Features

- Hero section with animated background
- Language selector cards with accent colors
- Sticky navbar showing current language and backend status
- Record button with timer and progress ring
- Upload button for audio files
- Waveform visualizer during capture
- Transcript panel with segment timestamps
- Confidence meter for STT output
- Text area with character counter for TTS
- Gender toggle for voice choice
- Speed and pitch sliders
- Audio player with seek bar and download action
- Session history section

## API Reference

### Health

- `GET /health`

Returns:

- backend status
- loaded STT model name
- TTS engine name
- supported language count

### Languages

- `GET /api/languages`

Returns metadata for the supported language keys:

- `english`
- `german`
- `spanish`
- `french`
- `nepali`

Each entry includes:

- label
- flag
- placeholder
- language code

### STT

- `POST /api/stt`
- `POST /api/transcribe`

Form fields:

- `audio`
- `language`
- `auto_detect`

Response fields:

- `transcript`
- `text`
- `language`
- `detected_language`
- `confidence`
- `duration_sec`
- `elapsed_sec`
- `segments`

### TTS

- `POST /api/tts`
- `POST /api/synthesize`

Form fields:

- `text`
- `language`
- `gender`
- `rate`
- `pitch`

Returns:

- MP3 audio stream

### Voices

- `GET /api/voices/{language}`

Returns:

- available voice names
- gender
- locale

## Model Details

### WhisperService

File:

- [backend/app/services/whisper_service.py](./backend/app/services/whisper_service.py)

Behavior:

- loads a Whisper model at startup
- can load a fine-tuned checkpoint when `FINETUNED_MODEL_DIR` is set
- supports automatic language detection or forced language codes

### EdgeTTSService

File:

- [backend/app/services/edge_tts_service.py](./backend/app/services/edge_tts_service.py)

Behavior:

- uses Microsoft neural voices
- supports all five app languages
- provides a simple runtime synthesis path without local GPU inference

### XTTSService

File:

- [backend/app/services/xtts_service.py](./backend/app/services/xtts_service.py)

Behavior:

- wraps Coqui XTTS-v2
- targets English, German, Spanish, French
- intentionally excludes Nepali because XTTS-v2 does not officially cover it in this repo design

### MMSASRService

File:

- [backend/app/services/mms_asr_service.py](./backend/app/services/mms_asr_service.py)

Behavior:

- wraps Meta MMS ASR
- acts as an alternative or comparison path for low-resource languages
- is especially relevant for Nepali

## Text and Audio Preprocessing

### Audio helpers

File:

- [backend/app/preprocessing/audio.py](./backend/app/preprocessing/audio.py)

What it does:

- creates temp paths
- loads audio
- resamples to 16 kHz
- converts stereo to mono
- trims silence
- saves WAV files

### Text normalization

File:

- [backend/app/preprocessing/text.py](./backend/app/preprocessing/text.py)

What it does:

- normalizes Unicode
- collapses whitespace
- expands a few English contractions
- cleans quotes in German and French
- preserves language-specific punctuation
- normalizes Nepali danda spacing

## Training Pipeline

### STT data preparation

Files:

- [data/download_common_voice.py](./data/download_common_voice.py)
- [training/stt/prepare_data.py](./training/stt/prepare_data.py)

Purpose:

- download Common Voice
- resample clips to 16 kHz
- normalize transcripts
- filter short or invalid samples
- write JSONL manifests for training and evaluation

### STT fine-tuning

Files:

- [training/stt/train_whisper.py](./training/stt/train_whisper.py)
- [backend/finetune/finetune_whisper.py](./backend/finetune/finetune_whisper.py)

Purpose:

- fine-tune Whisper on one language at a time
- use Common Voice or custom manifests
- report WER during training
- save the fine-tuned checkpoint for runtime use

### STT evaluation

File:

- [training/stt/eval_stt.py](./training/stt/eval_stt.py)

Purpose:

- compute WER and CER
- compare base vs fine-tuned checkpoints
- measure latency

### TTS data preparation

Files:

- [data/collect_tts_data.py](./data/collect_tts_data.py)
- [backend/finetune/prepare_custom_data.py](./backend/finetune/prepare_custom_data.py)

Purpose:

- validate custom recordings
- create labels/templates
- generate synthetic bootstrap samples
- convert raw audio into trainable manifests

### TTS fine-tuning

Files:

- [training/tts/finetune_xtts.py](./training/tts/finetune_xtts.py)
- [training/tts/finetune_nepali_tts.py](./training/tts/finetune_nepali_tts.py)

Purpose:

- fine-tune XTTS for EN/DE/ES/FR
- fine-tune MMS-TTS for Nepali
- save model checkpoints in `models/`

### TTS evaluation

File:

- [training/tts/eval_tts.py](./training/tts/eval_tts.py)

Purpose:

- measure synthesis latency
- run round-trip WER checks
- compare Edge TTS vs MMS-TTS for Nepali workflows

## Running The Project

### Docker

Recommended:

```bash
./docker-start.sh up
```

Useful commands:

- `./docker-start.sh logs`
- `./docker-start.sh restart`
- `./docker-start.sh down`
- `./docker-start.sh build`

Open:

- `http://localhost`

### Local development

```bash
./setup.sh
./run.sh
```

`setup.sh`:

- checks Python
- installs `ffmpeg` if needed
- creates `.venv`
- installs backend dependencies

`run.sh`:

- reuses `.venv` if present
- installs backend dependencies
- starts `uvicorn app.main:app`

## Environment Variables

See [.env.example](./.env.example).

Important values:

- `WHISPER_MODEL`
  - Whisper size to load
  - Examples: `tiny`, `base`, `small`, `medium`, `large-v3`

- `FINETUNED_MODEL_DIR`
  - Optional Whisper checkpoint directory

- `XTTS_MODEL_DIR`
  - Optional XTTS checkpoint directory

- `MMS_MODEL_DIR`
  - Optional MMS-TTS checkpoint directory

- `HF_TOKEN`
  - Required for Common Voice downloads

- `PORT`
  - Nginx host port when using Docker

## Important Design Notes

- The app is speech-first, not chat-first.
- There is no general-purpose local LLM in the runtime path.
- Open-source speech models are used where they fit best.
- The runtime model choice is intentionally lighter than adding a chat LLM.
- The repo includes enough training scaffolding to replace or improve the baseline models later.

## Troubleshooting

- If the app cannot transcribe, check `ffmpeg` and microphone permissions.
- If startup is slow, Whisper is probably loading the model.
- If Docker health checks take time, that is expected while models initialize.
- If Common Voice downloads fail, check `HF_TOKEN` and dataset access.
- If the frontend cannot talk to the backend, confirm same-origin routing or update `API` in `frontend/js/app.js`.

## License

No license file is present in the repository. Add one before distributing the project publicly.

- `FINETUNED_MODEL_DIR`
  - Optional path to a fine-tuned Whisper checkpoint

- `XTTS_MODEL_DIR`
  - Optional path to a fine-tuned XTTS checkpoint

- `MMS_MODEL_DIR`
  - Optional path to a fine-tuned Nepali MMS-TTS checkpoint

- `HF_TOKEN`
  - Needed for Hugging Face Common Voice downloads

- `PORT`
  - Nginx host port in Docker

## Running the App

### Docker

This is the recommended path.

```bash
./docker-start.sh up
```

Then open:

- `http://localhost`

Useful commands:

- `./docker-start.sh logs`
- `./docker-start.sh restart`
- `./docker-start.sh down`
- `./docker-start.sh build`

### Local Development

```bash
./setup.sh
./run.sh
```

`setup.sh`:

- checks for Python
- installs `ffmpeg` if missing
- creates `.venv`
- installs backend dependencies

`run.sh`:

- ensures `ffmpeg` exists
- creates or reuses `.venv`
- installs backend dependencies
- starts `uvicorn app.main:app`

## Training Pipeline

The repository includes training and preprocessing scripts for a real model workflow.

### STT data preparation

- [data/download_common_voice.py](./data/download_common_voice.py)
- [training/stt/prepare_data.py](./training/stt/prepare_data.py)

These scripts:

- download Mozilla Common Voice data
- normalize text
- resample audio to 16 kHz
- filter unusable clips
- write JSONL manifests for training

### STT training

- [training/stt/train_whisper.py](./training/stt/train_whisper.py)
- [backend/finetune/finetune_whisper.py](./backend/finetune/finetune_whisper.py)

These scripts fine-tune Whisper using:

- Common Voice data
- custom manifests
- optional Hugging Face tokens

### STT evaluation

- [training/stt/eval_stt.py](./training/stt/eval_stt.py)

Reports:

- WER
- CER
- average latency

### TTS data and training

- [data/collect_tts_data.py](./data/collect_tts_data.py)
- [backend/finetune/prepare_custom_data.py](./backend/finetune/prepare_custom_data.py)
- [training/tts/finetune_xtts.py](./training/tts/finetune_xtts.py)
- [training/tts/finetune_nepali_tts.py](./training/tts/finetune_nepali_tts.py)

The TTS pipeline supports:

- validating your own recorded voice data
- generating synthetic bootstrap samples
- preparing training manifests
- fine-tuning XTTS for EN/DE/ES/FR
- fine-tuning Nepali MMS-TTS for Nepali

## Audio and Text Processing

Audio utilities in [backend/app/preprocessing/audio.py](./backend/app/preprocessing/audio.py) handle:

- temp file creation
- resampling
- mono conversion
- normalization helpers
- silence trimming
- WAV saving and loading

Text normalization in [backend/app/preprocessing/text.py](./backend/app/preprocessing/text.py) handles:

- whitespace cleanup
- Unicode normalization
- language-specific punctuation cleanup
- Nepali danda spacing normalization

## Design Choices

### Why Whisper for STT

Whisper is a strong multilingual baseline and already fits the app’s five-language goal.

### Why Edge TTS for runtime TTS

Edge TTS is fast, free, and covers all five languages in the current app without requiring GPU inference at runtime.

### Why XTTS and MMS are still included

They are the upgrade path:

- XTTS for higher-quality multilingual TTS in EN/DE/ES/FR
- MMS for Nepali and other low-resource language paths

This is why the repo contains both the runtime app and the training infrastructure.

## Troubleshooting

- If transcription fails, check that `ffmpeg` is installed.
- If the backend is slow on first boot, Whisper is probably loading the model.
- If Docker health checks take a while, that is expected during model startup.
- If the frontend cannot reach the backend, confirm it is using the same origin or update `API` in `frontend/js/app.js`.
- If Common Voice downloads fail, confirm your `HF_TOKEN` is valid and the dataset terms are accepted.

## Notes

- The frontend is served from Nginx in Docker and from FastAPI in local mode.
- The modular backend in `backend/app/` is the current source of truth.
- The legacy `backend/main.py` file is still present for compatibility and reference.

## License

No explicit license file is present in the repository. Add one if you plan to publish or share the project.
