"""
Phase 7: Stage 2 - Sentiment analysis training with Seq2SeqTrainer
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
import re
from sklearn.metrics import accuracy_score, f1_score
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

def extract_sentiment(text):
    """Extract the sentiment label from the model output"""
    if not text:
        return "خنثی"

    # Different patterns for finding the sentiment
    patterns = [
        r'احساس:\s*([^\n]+)',
        r'احساس\s*:\s*([^\n]+)',
        r'احساسی:\s*([^\n]+)',
        # Fixed: was r'([مثبت|منفی|خنثی])' — square brackets matched only ONE letter,
        # so plain labels like "مثبت" were never recognized and everything became "neutral"
        r'(مثبت|منفی|خنثی)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            label = match.group(1).strip()
            if 'مثبت' in label:
                return 'مثبت'
            elif 'منفی' in label:
                return 'منفی'
            elif 'خنثی' in label:
                return 'خنثی'

    # If none was found, use the default label
    return "خنثی"

def compute_metrics(eval_pred):
    """Compute accuracy and F1 for sentiment analysis"""
    global tokenizer
    predictions, labels = eval_pred

    # ===== predictions are already token IDs (because predict_with_generate=True) =====
    # so argmax is not needed!

    # ===== Filter -100 from labels =====
    mask = labels != -100
    if not mask.any():
        return {'accuracy': 0.0, 'f1': 0.0}

    # ===== Convert to text =====
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 with pad_token_id for decoding
    labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    # ===== Extract labels =====
    pred_labels = [extract_sentiment(p) for p in decoded_preds]
    ref_labels = [extract_sentiment(r) for r in decoded_labels]

    # ===== Remove empty items =====
    valid = [i for i, (p, r) in enumerate(zip(pred_labels, ref_labels)) if p and r]
    if not valid:
        return {'accuracy': 0.0, 'f1': 0.0}

    preds = [pred_labels[i] for i in valid]
    refs = [ref_labels[i] for i in valid]

    accuracy = accuracy_score(refs, preds)
    f1 = f1_score(refs, preds, average='weighted', zero_division=0)

    return {'accuracy': accuracy, 'f1': f1}

def main():
    global tokenizer
    print_section("Stage 2: Sentiment Analysis Training (Seq2Seq)")

    # ===== Load dataset =====
    df = pd.read_csv(f"{config['data']['processed']}sent_clean.csv")
    print(f"📂 Total samples: {len(df):,}")

    print("\n📊 Class distribution:")
    print(df['label'].value_counts())

    # Sampling
    df = df.head(5000)
    print(f"⚠️ Using {len(df):,} samples")

    # ===== Prepare dataset =====
    dataset = Dataset.from_pandas(df[['clean_text', 'label']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # ===== Load model from Stage 1 (curriculum: summarization -> sentiment -> joint) =====
    model_path = f"{config['output']['models_dir']}summ_final"
    if not os.path.exists(model_path):
        print(f"❌ Stage 1 model not found: {model_path}")
        print("   Please run 4_train_stage1_summ.py first.")
        return
    print(f"🔄 Loading Stage 1 model (summarization): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)

    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"✅ Model on: {next(model.parameters()).device}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Preprocessing function =====
    def preprocess_function(examples):
        inputs = ["تحلیل احساس: " + text for text in examples['clean_text']]
        model_inputs = tokenizer(
            inputs,
            max_length=config['data']['max_input_length'],
            truncation=True,
            padding='max_length'
        )

        labels = tokenizer(
            examples['label'],
            max_length=20,
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
        output_dir=f"{config['output']['checkpoints_dir']}sent",
        num_train_epochs=config['model']['epochs_stage2'],
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=float(config['model']['learning_rate']),
        warmup_steps=config['model']['warmup_steps'],
        weight_decay=float(config['model']['weight_decay']),
        fp16=False,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        label_smoothing_factor=0.1,
        remove_unused_columns=False,
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=20,
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
    history_path = f"{config['output']['results_dir']}sent_training_history.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(history_path, index=False)
    print(f"✅ Training history saved: {history_path}")

    # ===== Save model =====
    save_path = f"{config['output']['models_dir']}sent_final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Model saved: {save_path}")

    print("\n📊 Final evaluation (validation split):")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f" {key}: {value:.4f}")

if __name__ == "__main__":
    main()


#     ### Old version: runs and produces output, but computes the evaluation metrics incorrectly

#     """
# Phase 7: Stage 2 - Sentiment analysis training with Seq2SeqTrainer
# """

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["JAX_PLATFORMS"] = "cpu"
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from transformers import (
#     MT5ForConditionalGeneration,
#     AutoTokenizer,
#     Seq2SeqTrainer,
#     Seq2SeqTrainingArguments,
#     EarlyStoppingCallback,
#     DataCollatorForSeq2Seq
# )
# from datasets import Dataset
# import pandas as pd
# import torch
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
# from utils import load_config, print_section

# # Clear GPU memory before starting
# torch.cuda.empty_cache()

# config = load_config()

# print(f"🔧 CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"🔧 GPU name: {torch.cuda.get_device_name(0)}")
#     print(f"🔧 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# os.makedirs(config['output']['models_dir'], exist_ok=True)
# os.makedirs(config['output']['checkpoints_dir'], exist_ok=True)
# os.makedirs(config['output']['results_dir'], exist_ok=True)

# tokenizer = None

# def compute_metrics(eval_pred):
#     """Compute accuracy and F1 for sentiment analysis"""
#     global tokenizer
#     predictions, labels = eval_pred
    
#     # Convert logits to token IDs
#     predictions = np.argmax(predictions, axis=-1)
    
#     # Filter -100 from labels
#     mask = labels != -100
#     if not mask.any():
#         return {'accuracy': 0.0, 'f1': 0.0}
    
#     # Convert to text
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
#     # Replace -100 with pad_token_id for decoding
#     labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    
#     # Remove empty items
#     valid = [i for i, (p, l) in enumerate(zip(decoded_preds, decoded_labels)) if p and l]
#     if not valid:
#         return {'accuracy': 0.0, 'f1': 0.0}
    
#     preds = [decoded_preds[i] for i in valid]
#     refs = [decoded_labels[i] for i in valid]
    
#     accuracy = accuracy_score(refs, preds)
#     f1 = f1_score(refs, preds, average='weighted', zero_division=0)
    
#     return {'accuracy': accuracy, 'f1': f1}

# def main():
#     global tokenizer
#     print_section("Stage 2: Sentiment Analysis Training (Seq2Seq)")
    
#     # ===== Load dataset =====
#     df = pd.read_csv(f"{config['data']['processed']}sent_clean.csv")
#     print(f"📂 Total samples: {len(df):,}")
    
#     print("\n📊 Class distribution:")
#     print(df['label'].value_counts())
    
#     # Sampling
#     df = df.head(3000)
#     print(f"⚠️ Using {len(df):,} samples")
    
#     # ===== Prepare dataset =====
#     dataset = Dataset.from_pandas(df[['clean_text', 'label']])
#     dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
#     # ===== Load model =====
#     print("🔄 Loading base model...")
#     tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])
#     model = MT5ForConditionalGeneration.from_pretrained(config['model']['base_model'])
    
#     if torch.cuda.is_available():
#         model = model.to('cuda')
#         print(f"✅ Model on: {next(model.parameters()).device}")
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # ===== Preprocessing function =====
#     def preprocess_function(examples):
#         inputs = ["تحلیل احساس: " + text for text in examples['clean_text']]
#         model_inputs = tokenizer(
#             inputs,
#             max_length=config['data']['max_input_length'],
#             truncation=True,
#             padding='max_length'
#         )
        
#         labels = tokenizer(
#             examples['label'],
#             max_length=20,
#             truncation=True,
#             padding='max_length'
#         )
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
    
#     print("🔄 Tokenizing data...")
#     tokenized_dataset = dataset.map(
#         preprocess_function,
#         batched=True,
#         remove_columns=dataset['train'].column_names
#     )
    
#     # ===== Data Collator =====
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         label_pad_token_id=-100,
#         pad_to_multiple_of=8 if torch.cuda.is_available() else None
#     )
    
#     # ===== Seq2Seq training settings =====
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=f"{config['output']['checkpoints_dir']}sent",
#         num_train_epochs=config['model']['epochs_stage2'],
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=2,
#         gradient_accumulation_steps=4,
#         learning_rate=float(config['model']['learning_rate']),
#         warmup_steps=config['model']['warmup_steps'],
#         weight_decay=float(config['model']['weight_decay']),
#         fp16=False,  # disabled to avoid NaN
#         eval_strategy="steps",
#         eval_steps=200,
#         save_steps=200,
#         save_total_limit=2,
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#         logging_steps=50,
#         report_to="none",
#         dataloader_num_workers=0,
#         optim="adamw_torch",
#         label_smoothing_factor=0.1,
#         remove_unused_columns=False,
#         prediction_loss_only=False,
#         predict_with_generate=True,
#         generation_max_length=20,
#         generation_num_beams=2,
#         dataloader_pin_memory=False,
#         max_grad_norm=1.0,
#     )
    
#     # ===== Trainer =====
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_dataset['train'],
#         eval_dataset=tokenized_dataset['test'],
#         compute_metrics=compute_metrics,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
#         data_collator=data_collator,
#     )
    
#     # ===== Training =====
#     print("🚀 Starting training...")
#     trainer.train()
    
#     # ===== Save model =====
#     save_path = f"{config['output']['models_dir']}sent_final"
#     model.save_pretrained(save_path)
#     tokenizer.save_pretrained(save_path)
#     print(f"✅ Model saved: {save_path}")
    
#     print("\n📊 Final evaluation:")
#     eval_results = trainer.evaluate()
#     for key, value in eval_results.items():
#         print(f" {key}: {value:.4f}")

# if __name__ == "__main__":
#     main()