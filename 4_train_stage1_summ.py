"""
Phase 7: Stage 1 - Summarization training with Seq2SeqTrainer  (fixed version)

Changes vs. previous version:
  1. Training uses ONLY the official PN-Summary train split (split == 'train').
     Validation uses a fixed sample of the official dev split.
     The official test split is never seen here (previously summ_clean.csv was
     re-split randomly, so test articles could end up in training).
  2. Random seed for sampling; no df.head().
  3. label_smoothing_factor = 0.0, so training and validation losses are plain
     cross-entropy on the same scale.
  4. Larger eval batch (generation during validation is faster).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import torch
import numpy as np
from rouge_score import rouge_scorer
from utils import load_config, print_section

# Clear GPU memory before starting
torch.cuda.empty_cache()

config = load_config()

print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU name: {torch.cuda.get_device_name(0)}")
    print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

os.makedirs(config['output']['models_dir'], exist_ok=True)
os.makedirs(config['output']['checkpoints_dir'], exist_ok=True)
os.makedirs(config['output']['results_dir'], exist_ok=True)

tokenizer = None
SEED = 42
VAL_SIZE = 500   # validation generates text at every eval step, so keep it moderate

class WhitespaceTokenizer:
    """Simple tokenizer for Persian ROUGE: split words on whitespace"""
    def tokenize(self, text):
        return text.split()

def compute_metrics(eval_pred):
    """Compute ROUGE for evaluation"""
    global tokenizer
    predictions, labels = eval_pred
    
    # ===== predictions are already token IDs (because predict_with_generate=True) =====
    # so argmax is not needed!
    
    # ===== Filter -100 from labels =====
    mask = labels != -100
    if not mask.any():
        return {'rouge1': 0.0}
    
    # ===== Convert to text =====
    # The Trainer pads generated predictions with -100; replace it before decoding
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 with pad_token_id for decoding
    labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    
    # ===== Compute ROUGE =====
    # The default rouge_score tokenizer keeps only English letters (a-z) and digits and
    # removes all Persian characters, which made ROUGE = 0. So we split text on whitespace.
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], tokenizer=WhitespaceTokenizer())
    rouge_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        if pred and ref:
            scores = scorer.score(pred, ref)
            rouge_scores.append(scores['rouge1'].fmeasure)
    
    return {'rouge1': np.mean(rouge_scores) if rouge_scores else 0.0}

def main():
    global tokenizer
    print_section("Stage 1: Summarization Training (Seq2Seq)")
    
    # ===== Load dataset =====
    df = pd.read_csv(f"{config['data']['processed']}summ_clean.csv")
    df = df.dropna(subset=['clean_text', 'clean_summary'])
    print(f"📂 Total samples: {len(df):,}")

    if 'split' not in df.columns:
        raise ValueError("summ_clean.csv has no 'split' column. Run the fixed 2_preprocess.py first.")

    # Official splits only: train for training, a fixed dev sample for validation.
    # The official test split is NOT used here.
    train_df = df[df['split'] == 'train']
    train_df = train_df[train_df['clean_summary'].str.len() > 10]
    # Remove any train/dev article that also appears in the official test split (no leakage)
    test_texts = set(df.loc[df['split'] == 'test', 'clean_text'])
    train_df = train_df[~train_df['clean_text'].isin(test_texts)].drop_duplicates(subset='clean_text')
    df = df[(df['split'] == 'test') | (~df['clean_text'].isin(test_texts))]
    val_df = df[df['split'] == 'dev'].sample(n=min(VAL_SIZE, (df['split'] == 'dev').sum()), random_state=SEED)
    print(f"   Train (official train): {len(train_df):,}")
    print(f"   Validation (official dev sample): {len(val_df):,}")
    print(f"   Official test (not used here): {(df['split'] == 'test').sum():,}")

    if config.get('smoke_rows'):                     # smoke test only
        train_df, val_df = train_df.head(config['smoke_rows']), val_df.head(config['smoke_rows'])

    # ===== Prepare dataset =====
    from datasets import DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df[['clean_text', 'clean_summary']], preserve_index=False),
        'test': Dataset.from_pandas(val_df[['clean_text', 'clean_summary']], preserve_index=False),
    })
    
    # ===== Load model =====
    print("🔄 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
    model = MT5ForConditionalGeneration.from_pretrained(config['model']['base_model'])
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"✅ Model on: {next(model.parameters()).device}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== Preprocessing function =====
    def preprocess_function(examples):
        inputs = ["خلاصه کن: " + text for text in examples['clean_text']]
        model_inputs = tokenizer(
            inputs,
            max_length=config['data']['max_input_length'],
            truncation=True,
            padding='max_length'
        )
        
        labels = tokenizer(
            examples['clean_summary'],
            max_length=config['data']['max_summary_length'],
            truncation=True,
            padding='max_length'
        )
        # Replace pad tokens in the labels with -100 so they are ignored in the loss
        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels["input_ids"]
        ]
        return model_inputs
    
    print("🔄 Tokenizing data...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # ===== Data Collator =====
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    
    # ===== Seq2Seq training settings =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{config['output']['checkpoints_dir']}summ",
        num_train_epochs=config['model']['epochs_stage1'],
        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=float(config['model']['learning_rate']),
        warmup_steps=config['model']['warmup_steps'],
        weight_decay=float(config['model']['weight_decay']),
        fp16=False,
        eval_strategy="steps",
        eval_steps=2 if config.get('smoke_rows') else 600,   # each evaluation generates 500 summaries (~6 min)
        save_steps=2 if config.get('smoke_rows') else 600,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        label_smoothing_factor=0.0,   # plain cross-entropy: train/val losses on the same scale
        seed=SEED,
        remove_unused_columns=False,
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=config['data']['max_summary_length'],
        generation_num_beams=2,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
    )
    
    # ===== Trainer =====
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        data_collator=data_collator,
    )
    
    # ===== Training =====
    print("🚀 Starting training...")
    trainer.train()

    # ===== Save training history (for the loss-curve plot in step 8) =====
    history_path = f"{config['output']['results_dir']}summ_training_history.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(history_path, index=False)
    print(f"✅ Training history saved: {history_path}")
    
    # ===== Save model =====
    save_path = f"{config['output']['models_dir']}summ_final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    import shutil
    shutil.rmtree(f"{config['output']['checkpoints_dir']}summ", ignore_errors=True)   # free disk space
    print(f"✅ Model saved: {save_path}")
    
    print("\n📊 Final evaluation (validation split):")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f" {key}: {value:.4f}")

if __name__ == "__main__":
    main()

