"""
فاز ۹: ارزیابی کامل مدل نهایی (فقط مدل مشترک)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import MT5ForConditionalGeneration, AutoTokenizer
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
from utils import load_config, print_section, save_json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

config = load_config()

print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU name: {torch.cuda.get_device_name(0)}")

os.makedirs(config['output']['results_dir'], exist_ok=True)

def extract_sentiment(text):
    """استخراج برچسب احساس از خروجی مدل"""
    if not text:
        return "خنثی"
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
    return "خنثی"

def extract_summary(text):
    """استخراج خلاصه از خروجی مدل"""
    if not text:
        return ""
    # حذف بخش احساس
    summary = re.sub(r'\n?احساس:\s*[^\n]+', '', text)
    summary = re.sub(r'خلاصه:\s*', '', summary)
    return summary.strip()

def main():
    print_section("فاز ۹: ارزیابی کامل مدل نهایی")
    
    # ===== بارگذاری دیتاست تست =====
    df_test = pd.read_csv(f"{config['data']['processed']}joint_test.csv")
    print(f"📂 تعداد کل نمونه‌ها: {len(df_test):,}")
    
    #  - می‌توانید تعداد را افزایش دهید
    df_test = df_test.head(2000)
    print(f"⚠️ برای تست سریع فقط {len(df_test):,} نمونه استفاده می‌شود")
    
    # ===== بارگذاری مدل مشترک =====
    model_path = f"{config['output']['models_dir']}joint_final"
    
    if not os.path.exists(model_path):
        print(f"❌ مدل در {model_path} وجود ندارد!")
        print("   لطفاً ابتدا مرحله ۳ را اجرا کنید.")
        return
    
    print("🔄 بارگذاری مدل مشترک...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== تولید پیش‌بینی‌ها =====
    texts = df_test['text'].tolist()
    references = df_test['target'].tolist()
    
    print("\n🔄 تولید پیش‌بینی‌ها...")
    predictions = []
    for text in tqdm(texts, desc="   تولید"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        predictions.append(pred)
    
    # ===== ذخیره نتایج =====
    results_df = pd.DataFrame({
        'text': texts,
        'reference': references,
        'prediction': predictions
    })
    results_df.to_csv(f"{config['output']['results_dir']}predictions.csv", index=False)
    print("✅ پیش‌بینی‌ها ذخیره شدند")
    
    # ===== ارزیابی خلاصه‌سازی =====
    print_section("ارزیابی خلاصه‌سازی")
    
    pred_summaries = [extract_summary(p) for p in predictions]
    ref_summaries = [extract_summary(r) for r in references]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in tqdm(zip(pred_summaries, ref_summaries), total=len(pred_summaries), desc="   محاسبه ROUGE"):
        if pred and ref:
            scores = scorer.score(pred, ref)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
    
    avg_rouge = {
        'rouge1': np.mean([s['rouge1'] for s in rouge_scores]) if rouge_scores else 0,
        'rouge2': np.mean([s['rouge2'] for s in rouge_scores]) if rouge_scores else 0,
        'rougeL': np.mean([s['rougeL'] for s in rouge_scores]) if rouge_scores else 0
    }
    
    print(f"\n📊 ROUGE Scores:")
    print(f"   ROUGE-1: {avg_rouge['rouge1']:.4f}")
    print(f"   ROUGE-2: {avg_rouge['rouge2']:.4f}")
    print(f"   ROUGE-L: {avg_rouge['rougeL']:.4f}")
    
    # ===== ارزیابی تحلیل احساس =====
    print_section("ارزیابی تحلیل احساس")
    
    pred_labels = [extract_sentiment(p) for p in predictions]
    ref_labels = [extract_sentiment(r) for r in references]
    
    accuracy = accuracy_score(ref_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ref_labels, pred_labels, average='weighted', zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(ref_labels, pred_labels, average='macro', zero_division=0)[2]
    cm = confusion_matrix(ref_labels, pred_labels, labels=['مثبت', 'خنثی', 'منفی'])
    
    print(f"\n📊 نتایج تحلیل احساس:")
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ Precision (Weighted): {precision:.4f}")
    print(f"   ✅ Recall (Weighted): {recall:.4f}")
    print(f"   ✅ F1 (Weighted): {f1:.4f}")
    print(f"   ✅ F1 (Macro): {macro_f1:.4f}")
    
    print(f"\n📋 گزارش طبقه‌بندی:")
    print(classification_report(ref_labels, pred_labels, target_names=['مثبت', 'خنثی', 'منفی'], zero_division=0))
    
    print(f"\n📊 ماتریس درهم‌ریختگی:")
    cm_df = pd.DataFrame(cm, index=['مثبت', 'خنثی', 'منفی'], columns=['مثبت', 'خنثی', 'منفی'])
    print(cm_df)
    
    # ===== ذخیره نتایج نهایی =====
    final_results = {
        'summary': avg_rouge,
        'sentiment': {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'f1_macro': macro_f1,
            'confusion_matrix': cm.tolist()
        },
        'num_samples': len(texts),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    save_json(final_results, f"{config['output']['results_dir']}evaluation_results.json")
    print(f"\n✅ نتایج ارزیابی ذخیره شد: {config['output']['results_dir']}evaluation_results.json")

if __name__ == "__main__":
    main()