"""
فاز ۷: مرحله ۳ - آموزش مشترک (Joint Learning) با Seq2SeqTrainer
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
import re  # ← اضافه شد
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from utils import load_config, print_section

# پاکسازی حافظه GPU قبل از شروع
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
    """استخراج برچسب احساس از خروجی مدل"""
    if not text:
        return "خنثی"
    
    # الگوهای مختلف برای پیدا کردن احساس
    patterns = [
        r'احساس:\s*([^\n]+)',
        r'احساس\s*:\s*([^\n]+)',
        r'احساسی:\s*([^\n]+)',
        r'([مثبت|منفی|خنثی])',
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
    
    # اگر هیچکدام پیدا نشد، برچسب پیش‌فرض
    return "خنثی"

def compute_metrics(eval_pred):
    """محاسبه ROUGE + دقت و F1 برای ارزیابی مشترک"""
    global tokenizer
    predictions, labels = eval_pred
    
    # ===== predictions قبلاً token ID هستند (به دلیل predict_with_generate=True) =====
    # بنابراین نیازی به argmax نیست!
    
    # ===== فیلتر کردن -100 از labels =====
    mask = labels != -100
    if not mask.any():
        return {'rouge1': 0.0, 'accuracy': 0.0, 'f1': 0.0}
    
    # ===== تبدیل به متن =====
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # جایگزینی -100 با pad_token_id برای decode
    labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    
    # ===== ROUGE برای خلاصه‌سازی =====
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        if pred and ref:
            # حذف بخش احساس از متن برای محاسبه ROUGE
            pred_clean = pred
            ref_clean = ref
            if 'احساس:' in pred:
                pred_clean = pred.split('احساس:')[0].replace('خلاصه:', '').strip()
            if 'احساس:' in ref:
                ref_clean = ref.split('احساس:')[0].replace('خلاصه:', '').strip()
            if pred_clean and ref_clean:
                scores = scorer.score(pred_clean, ref_clean)
                rouge_scores.append(scores['rouge1'].fmeasure)
    
    # ===== دقت و F1 برای تحلیل احساس =====
    pred_labels = [extract_sentiment(p) for p in decoded_preds]
    ref_labels = [extract_sentiment(r) for r in decoded_labels]
    
    valid = [i for i, (p, r) in enumerate(zip(pred_labels, ref_labels)) if p and r]
    if valid:
        preds = [pred_labels[i] for i in valid]
        refs = [ref_labels[i] for i in valid]
        accuracy = accuracy_score(refs, preds)
        f1 = f1_score(refs, preds, average='weighted', zero_division=0)
    else:
        accuracy = 0.0
        f1 = 0.0
    
    return {
        'rouge1': np.mean(rouge_scores) if rouge_scores else 0.0,
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    global tokenizer
    print_section("فاز ۷ - مرحله ۳: آموزش مشترک (Seq2Seq)")
    
    # ===== بارگذاری دیتاست =====
    df = pd.read_csv(f"{config['data']['processed']}joint_clean.csv")
    print(f"📂 تعداد کل نمونه‌ها: {len(df):,}")
    
    # نمونه‌برداری
    df = df.head(5000)
    print(f"⚠️ برای تست فقط {len(df):,} نمونه استفاده می‌شود")
    
    # ===== آماده‌سازی دیتاست =====
    dataset = Dataset.from_pandas(df[['text', 'target']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # ===== بارگذاری مدل از مرحله ۱ =====
    print("🔄 بارگذاری مدل از مرحله ۱ (خلاصه‌سازی)...")
    model_path = f"{config['output']['models_dir']}summ_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    print(f"✅ مدل بارگذاری شد: {model_path}")
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        print(f"✅ Model on: {next(model.parameters()).device}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== تابع پیش‌پردازش =====
    def preprocess_function(examples):
        inputs = ["نظر: " + text for text in examples['text']]
        model_inputs = tokenizer(
            inputs,
            max_length=config['data']['max_input_length'],
            truncation=True,
            padding='max_length'
        )
        
        labels = tokenizer(
            examples['target'],
            max_length=config['data']['max_summary_length'] + 30,
            truncation=True,
            padding='max_length'
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print("🔄 توکنایز کردن داده‌ها...")
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
    
    # ===== تنظیمات آموزش با Seq2Seq =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{config['output']['checkpoints_dir']}joint",
        num_train_epochs=config['model']['epochs_joint'],
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=float(config['model']['learning_rate']) * 0.3,
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
        generation_max_length=config['data']['max_summary_length'] + 30,
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
    
    # ===== آموزش =====
    print("🚀 شروع آموزش مشترک...")
    trainer.train()
    
    # ===== ذخیره مدل =====
    save_path = f"{config['output']['models_dir']}joint_final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ مدل نهایی ذخیره شد: {save_path}")
    
    print("\n📊 ارزیابی نهایی روی مجموعه تست:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f" {key}: {value:.4f}")

if __name__ == "__main__":
    main()


    #######کدی که مشکل ارزیابی داره 
#     """
# فاز ۷: مرحله ۳ - آموزش مشترک (Joint Learning) با Seq2SeqTrainer
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
# from rouge_score import rouge_scorer
# from sklearn.metrics import accuracy_score, f1_score
# from utils import load_config, print_section

# # پاکسازی حافظه GPU قبل از شروع
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

# def extract_sentiment(text):
#     """استخراج برچسب احساس از خروجی مدل"""
#     if not text:
#         return ""
#     if 'احساس:' in text:
#         parts = text.split('احساس:')
#         if len(parts) > 1:
#             label = parts[1].strip().split()[0] if parts[1].strip() else ""
#             if 'مثبت' in label:
#                 return 'مثبت'
#             elif 'منفی' in label:
#                 return 'منفی'
#             elif 'خنثی' in label:
#                 return 'خنثی'
#     return ""

# def compute_metrics(eval_pred):
#     """محاسبه ROUGE + دقت و F1 برای ارزیابی مشترک"""
#     global tokenizer
#     predictions, labels = eval_pred
    
#     # تبدیل logits به token IDs
#     predictions = np.argmax(predictions, axis=-1)
    
#     # فیلتر کردن -100 از labels
#     mask = labels != -100
#     if not mask.any():
#         return {'rouge1': 0.0, 'accuracy': 0.0, 'f1': 0.0}
    
#     # تبدیل به متن
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
#     # جایگزینی -100 با pad_token_id برای decode
#     labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
    
#     # ===== ROUGE برای خلاصه‌سازی =====
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     rouge_scores = []
#     for pred, ref in zip(decoded_preds, decoded_labels):
#         if pred and ref:
#             # حذف بخش احساس از متن برای محاسبه ROUGE
#             pred_clean = pred.split('احساس:')[0].replace('خلاصه:', '').strip()
#             ref_clean = ref.split('احساس:')[0].replace('خلاصه:', '').strip()
#             if pred_clean and ref_clean:
#                 scores = scorer.score(pred_clean, ref_clean)
#                 rouge_scores.append(scores['rouge1'].fmeasure)
    
#     # ===== دقت و F1 برای تحلیل احساس =====
#     pred_labels = [extract_sentiment(p) for p in decoded_preds]
#     ref_labels = [extract_sentiment(r) for r in decoded_labels]
    
#     valid = [i for i, (p, r) in enumerate(zip(pred_labels, ref_labels)) if p and r]
#     if valid:
#         preds = [pred_labels[i] for i in valid]
#         refs = [ref_labels[i] for i in valid]
#         accuracy = accuracy_score(refs, preds)
#         f1 = f1_score(refs, preds, average='weighted', zero_division=0)
#     else:
#         accuracy = 0.0
#         f1 = 0.0
    
#     return {
#         'rouge1': np.mean(rouge_scores) if rouge_scores else 0.0,
#         'accuracy': accuracy,
#         'f1': f1
#     }

# def main():
#     global tokenizer
#     print_section("فاز ۷ - مرحله ۳: آموزش مشترک (Seq2Seq)")
    
#     # ===== بارگذاری دیتاست =====
#     df = pd.read_csv(f"{config['data']['processed']}joint_clean.csv")
#     print(f"📂 تعداد کل نمونه‌ها: {len(df):,}")
    
#     # نمونه‌برداری
#     df = df.head(3000)
#     print(f"⚠️ برای تست فقط {len(df):,} نمونه استفاده می‌شود")
    
#     # ===== آماده‌سازی دیتاست =====
#     dataset = Dataset.from_pandas(df[['text', 'target']])
#     dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
#     # ===== بارگذاری مدل از مرحله ۱ =====
#     print("🔄 بارگذاری مدل از مرحله ۱ (خلاصه‌سازی)...")
#     model_path = f"{config['output']['models_dir']}summ_final"
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = MT5ForConditionalGeneration.from_pretrained(model_path)
#     print(f"✅ مدل بارگذاری شد: {model_path}")
    
#     if torch.cuda.is_available():
#         model = model.to('cuda')
#         print(f"✅ Model on: {next(model.parameters()).device}")
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # ===== تابع پیش‌پردازش =====
#     def preprocess_function(examples):
#         inputs = ["نظر: " + text for text in examples['text']]
#         model_inputs = tokenizer(
#             inputs,
#             max_length=config['data']['max_input_length'],
#             truncation=True,
#             padding='max_length'
#         )
        
#         labels = tokenizer(
#             examples['target'],
#             max_length=config['data']['max_summary_length'] + 30,
#             truncation=True,
#             padding='max_length'
#         )
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs
    
#     print("🔄 توکنایز کردن داده‌ها...")
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
    
#     # ===== تنظیمات آموزش با Seq2Seq =====
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=f"{config['output']['checkpoints_dir']}joint",
#         num_train_epochs=config['model']['epochs_joint'],
#         per_device_train_batch_size=2,
#         per_device_eval_batch_size=2,
#         gradient_accumulation_steps=4,
#         learning_rate=float(config['model']['learning_rate']) * 0.3,
#         warmup_steps=config['model']['warmup_steps'],
#         weight_decay=float(config['model']['weight_decay']),
#         fp16=False,  # غیرفعال برای جلوگیری از NaN
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
#         generation_max_length=config['data']['max_summary_length'] + 30,
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
    
#     # ===== آموزش =====
#     print("🚀 شروع آموزش مشترک...")
#     trainer.train()
    
#     # ===== ذخیره مدل =====
#     save_path = f"{config['output']['models_dir']}joint_final"
#     model.save_pretrained(save_path)
#     tokenizer.save_pretrained(save_path)
#     print(f"✅ مدل نهایی ذخیره شد: {save_path}")
    
#     print("\n📊 ارزیابی نهایی روی مجموعه تست:")
#     eval_results = trainer.evaluate()
#     for key, value in eval_results.items():
#         print(f" {key}: {value:.4f}")

# if __name__ == "__main__":
#     main()