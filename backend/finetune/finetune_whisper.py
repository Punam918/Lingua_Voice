"""
Fine-tune OpenAI Whisper on Mozilla Common Voice for any of the 5 supported languages.

Usage:
    python finetune_whisper.py --language nepali --model_size small --epochs 3

Datasets used (Mozilla Common Voice 11.0 via HuggingFace):
    https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0

Requirements (install separately):
    pip install -r requirements_finetune.txt
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ─── Language Config ──────────────────────────────────────────────────────────

LANGUAGE_CONFIG = {
    "english": {"code": "en", "cv_name": "en", "task": "transcribe"},
    "german":  {"code": "de", "cv_name": "de", "task": "transcribe"},
    "spanish": {"code": "es", "cv_name": "es", "task": "transcribe"},
    "french":  {"code": "fr", "cv_name": "fr", "task": "transcribe"},
    "nepali":  {"code": "ne", "cv_name": "ne", "task": "transcribe"},
}

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


# ─── Data Collator ────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be different lengths
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it's ignored in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present (it's appended later)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    cfg = LANGUAGE_CONFIG[args.language]
    lang_code = cfg["code"]
    model_id = f"openai/whisper-{args.model_size}"
    output_dir = f"./whisper-{args.model_size}-{args.language}"

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {model_id}")
    print(f"  Language:    {args.language} ({lang_code})")
    print(f"  Output:      {output_dir}")
    print(f"{'='*60}\n")

    # ── Load Components ──────────────────────────────────────────────────────
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language=lang_code, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_id, language=lang_code, task="transcribe")

    # ── Load Dataset ─────────────────────────────────────────────────────────
    print(f"Loading Mozilla Common Voice 11.0 [{lang_code}]…")
    ds = DatasetDict()
    ds["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        cfg["cv_name"],
        split=f"train+validation",
        use_auth_token=args.hf_token,
        trust_remote_code=True,
    )
    ds["test"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        cfg["cv_name"],
        split="test",
        use_auth_token=args.hf_token,
        trust_remote_code=True,
    )

    # Keep only audio + sentence columns
    keep_cols = ["audio", "sentence"]
    remove_cols = [c for c in ds["train"].column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)

    # Resample to 16 kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # ── Preprocessing ────────────────────────────────────────────────────────
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Preprocessing dataset…")
    ds = ds.map(
        prepare_dataset,
        remove_columns=ds.column_names["train"],
        num_proc=args.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.gradient_checkpointing:
        model.config.use_cache = False

    # ── Metrics ──────────────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # ── Training Arguments ───────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=500,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=f"whisper-{args.model_size}-{args.language}" if args.push_to_hub else None,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("Starting training…")
    trainer.train()

    print(f"\nSaving fine-tuned model to: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("\nDone! Use the fine-tuned model in main.py by setting:")
    print(f'    stt_model = whisper.load_model("{output_dir}")')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Mozilla Common Voice")
    parser.add_argument("--language",    choices=list(LANGUAGE_CONFIG), default="nepali",
                        help="Language to fine-tune on")
    parser.add_argument("--model_size",  choices=WHISPER_MODELS,        default="small",
                        help="Whisper model size")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--max_steps",   type=int,   default=4000,
                        help="Max training steps (overrides epochs if set)")
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--grad_accum",  type=int,   default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--save_steps",  type=int,   default=500)
    parser.add_argument("--eval_steps",  type=int,   default=500)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Upload fine-tuned model to HuggingFace Hub")
    parser.add_argument("--hf_token",   type=str,   default=None,
                        help="HuggingFace token (needed for Common Voice download)")
    args = parser.parse_args()
    main(args)
