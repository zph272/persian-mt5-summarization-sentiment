"""
Shared training code for Stage 2, Stage 3 and the ablation runs.

- Data come ONLY from joint_train.csv / joint_val.csv (made by 3_prepare_joint_data.py).
  The test split (joint_test.csv) is never used for training.
- Inputs already contain the task prefix ("خلاصه کن: " or "احساس: ").
- Padding in the targets is masked with -100 (ignored in the loss).
- label_smoothing = 0, so training and validation losses are plain cross-entropy.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (MT5ForConditionalGeneration, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq)
from utils import load_config

config = load_config()
SEED = 42
LABELS = ["مثبت", "خنثی", "منفی"]
INVALID = "نامعتبر"

MODELS_DIR = config['output']['models_dir']
RESULTS_DIR = config['output']['results_dir']
CKPT_DIR = config['output']['checkpoints_dir']
for d in (MODELS_DIR, RESULTS_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)


def parse_sentiment(text):
    """Map a generated string to one label. Anything that is not exactly one label -> INVALID (counted as an error)."""
    t = (text or "").replace("احساس:", "").strip(" .:،\n")
    if t in LABELS:
        return t
    found = [lab for lab in LABELS if lab in t]
    return found[0] if len(found) == 1 else INVALID


def model_path(name):
    """'base' -> the pretrained mT5 from config; anything else -> a folder in outputs/models/."""
    if name == 'base':
        return config['model']['base_model']
    path = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model not found: {path}. Run the previous stage first.")
    return path


def load_split(split, task=None):
    """Read joint_<split>.csv; optionally keep only one task ('summ' or 'sent')."""
    df = pd.read_csv(f"{config['data']['processed']}joint_{split}.csv", keep_default_na=False)
    if task:
        df = df[df['task'] == task]
    return df.reset_index(drop=True)


def print_gpu():
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")


class PerExampleLossTrainer(Seq2SeqTrainer):
    """
    Loss = mean over examples of (mean cross-entropy over that example's target tokens).
    With the default token-level mean, a 60-token summary weighs ~20x more than a 3-token
    sentiment label, so in Stage 3 sentiment would contribute only ~1% of the loss.
    Here every example has the same weight, whatever the length of its target.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False      # Trainer then divides by gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})   # decoder_input_ids come from the collator
        logits = outputs.logits.float()
        tok = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                              ignore_index=-100, reduction="none").view(labels.shape)
        mask = (labels != -100).float()
        per_example = (tok * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        loss = per_example.mean()
        return (loss, outputs) if return_outputs else loss


def train_seq2seq(train_df, val_df, init, name, epochs, eval_steps,
                  max_target_len, sentiment_metrics=False, per_example_loss=False):
    """
    Fine-tune an mT5 model.
      init  : 'base' or the name of a saved model (e.g. 'summ_final')
      name  : output name (saved to outputs/models/<name>)
      sentiment_metrics : if True, validation also generates labels and reports accuracy / macro-F1
      per_example_loss  : if True, every example has equal weight in the loss (used in Stage 3)
    """
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    if config.get('smoke_rows'):                     # smoke test only
        n = config['smoke_rows']
        train_df, val_df, eval_steps = train_df.head(n), val_df.head(n), 2
    path = model_path(init)
    print(f"🔄 Loading model from: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = MT5ForConditionalGeneration.from_pretrained(path)
    if torch.cuda.is_available():
        model = model.to('cuda')

    print(f"   Train: {len(train_df):,}   Validation: {len(val_df):,}")
    if 'task' in train_df:
        print(f"   Train by task: {train_df['task'].value_counts().to_dict()}")
    print(f"   Example input : {train_df.iloc[0]['text'][:90]}...")
    print(f"   Example target: {train_df.iloc[0]['target'][:90]}")

    def tok(batch):
        enc = tokenizer(batch['text'], max_length=config['data']['max_input_length'], truncation=True)
        lab = tokenizer(text_target=batch['target'], max_length=max_target_len, truncation=True)
        enc['labels'] = lab['input_ids']          # dynamic padding: the collator pads labels with -100
        return enc

    train_ds = Dataset.from_pandas(train_df[['text', 'target']]).map(tok, batched=True, remove_columns=['text', 'target'])
    val_ds = Dataset.from_pandas(val_df[['text', 'target']]).map(tok, batched=True, remove_columns=['text', 'target'])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100,
                                      pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        p = [parse_sentiment(x) for x in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        r = [parse_sentiment(x) for x in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {'accuracy': accuracy_score(r, p),
                'macro_f1': f1_score(r, p, labels=LABELS, average='macro', zero_division=0)}

    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(CKPT_DIR, name),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,                       # effective batch size 8
        learning_rate=float(config['model']['learning_rate']),
        warmup_steps=config['model']['warmup_steps'],
        weight_decay=float(config['model']['weight_decay']),
        fp16=False,                                          # mT5 is unstable in fp16
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        label_smoothing_factor=0.0,
        remove_unused_columns=False,
        predict_with_generate=sentiment_metrics,             # generation only for short sentiment labels
        prediction_loss_only=not sentiment_metrics,
        generation_max_length=8,
        generation_num_beams=1,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        seed=SEED,
    )

    trainer_cls = PerExampleLossTrainer if per_example_loss else Seq2SeqTrainer
    trainer = trainer_cls(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if sentiment_metrics else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("🚀 Starting training...")
    t0 = time.time()
    trainer.train()
    minutes = round((time.time() - t0) / 60, 1)

    hist_path = f"{RESULTS_DIR}{name}_training_history.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(hist_path, index=False)

    save_path = os.path.join(MODELS_DIR, name)
    trainer.save_model(save_path)              # the best checkpoint (lowest validation loss)
    tokenizer.save_pretrained(save_path)

    final = trainer.evaluate()
    meta = {'name': name, 'init': init, 'per_example_loss': per_example_loss, 'n_train': len(train_df), 'n_val': len(val_df),
            'epochs_max': epochs, 'steps_done': trainer.state.global_step,
            'best_checkpoint': trainer.state.best_model_checkpoint,
            'best_val_metrics': {k: float(v) for k, v in final.items()},
            'train_minutes': minutes,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    with open(f"{RESULTS_DIR}{name}_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    shutil.rmtree(args.output_dir, ignore_errors=True)    # free disk space (checkpoints are large)
    print(f"✅ Model saved: {save_path}   ({minutes} min)")
    print(f"✅ History: {hist_path}")
    print("📊 Validation (best model):", {k: round(float(v), 4) for k, v in final.items()})
