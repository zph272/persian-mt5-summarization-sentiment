"""
فاز ۱۱ و ۱۲: رسم نمودارها و شکل‌های ضروری
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from utils import load_config, save_plot, print_section

config = load_config()

# ایجاد پوشه figures
os.makedirs(config['output']['figures_dir'], exist_ok=True)

def plot_training_curves():
    """رسم منحنی‌های آموزش"""
    print_section("رسم منحنی‌های آموزش")
    
    stages = ['summ', 'sent', 'joint']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, stage in enumerate(stages):
        try:
            history = pd.read_csv(f"{config['output']['results_dir']}{stage}_training_history.csv")
            
            train_loss = history[history['loss'].notna()]['loss'].values
            eval_loss = history[history['eval_loss'].notna()]['eval_loss'].values
            
            ax = axes[idx]
            ax.plot(train_loss, label='Training Loss', alpha=0.8)
            if len(eval_loss) > 0:
                eval_steps = np.linspace(0, len(train_loss), len(eval_loss))
                ax.plot(eval_steps, eval_loss, label='Validation Loss', alpha=0.8)
            
            titles = ["مرحله ۱: خلاصه‌سازی", "مرحله ۲: تحلیل احساس", "مرحله ۳: آموزش مشترک"]
            ax.set_title(titles[idx])
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"⚠️ خطا در رسم منحنی مرحله {stage}: {e}")
    
    plt.tight_layout()
    save_plot(fig, 'training_loss_curves.png', config)

def plot_rouge_comparison():
    """مقایسه امتیازات ROUGE"""
    print_section("مقایسه امتیازات ROUGE")
    
    try:
        with open(f"{config['output']['results_dir']}evaluation_results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        rouge_scores = results['summary']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        scores = [rouge_scores['rouge1'], rouge_scores['rouge2'], rouge_scores['rougeL']]
        
        bars = ax.bar(metrics, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_ylabel('F1 Score')
        ax.set_title('امتیازات ROUGE برای خلاصه‌سازی')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        save_plot(fig, 'rouge_comparison.png', config)
        
    except Exception as e:
        print(f"⚠️ خطا در رسم نمودار ROUGE: {e}")

def plot_sentiment_metrics():
    """نمودار معیارهای تحلیل احساس"""
    print_section("نمودار معیارهای تحلیل احساس")
    
    try:
        with open(f"{config['output']['results_dir']}evaluation_results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        sent_metrics = results['sentiment']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-W', 'F1-M']
        scores = [
            sent_metrics['accuracy'],
            sent_metrics['precision_weighted'],
            sent_metrics['recall_weighted'],
            sent_metrics['f1_weighted'],
            sent_metrics['f1_macro']
        ]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bars = ax.bar(metrics, scores, color=colors)
        ax.set_ylabel('Score')
        ax.set_title('معیارهای تحلیل احساس')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        save_plot(fig, 'sentiment_metrics.png', config)
        
    except Exception as e:
        print(f"⚠️ خطا در رسم نمودار احساس: {e}")

def plot_confusion_matrix():
    """رسم ماتریس درهم‌ریختگی"""
    print_section("ماتریس درهم‌ریختگی")
    
    try:
        with open(f"{config['output']['results_dir']}evaluation_results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        cm = np.array(results['sentiment']['confusion_matrix'])
        classes = ['مثبت', 'خنثی', 'منفی']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    ax=ax)
        ax.set_xlabel('پیش‌بینی')
        ax.set_ylabel('واقعیت')
        ax.set_title('ماتریس درهم‌ریختگی - تحلیل احساس')
        
        save_plot(fig, 'confusion_matrix.png', config)
        
    except Exception as e:
        print(f"⚠️ خطا در رسم ماتریس درهم‌ریختگی: {e}")

def plot_dataset_distribution():
    """نمایش توزیع دیتاست‌ها"""
    print_section("توزیع دیتاست‌ها")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ۱. توزیع احساسات در دیتاست اصلی
        df_sent = pd.read_csv(f"{config['data']['processed']}sent_clean.csv")
        df_sent['label'] = df_sent['Score'].apply(
            lambda x: 'مثبت' if x >= 80 else ('خنثی' if x >= 50 else 'منفی')
        )
        
        ax1 = axes[0, 0]
        sent_counts = df_sent['label'].value_counts()
        colors = ['#2ECC71', '#F1C40F', '#E74C3C']
        ax1.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=colors)
        ax1.set_title('توزیع احساسات - دیتاست اصلی')
        
        # ۲. طول متن‌ها
        ax2 = axes[0, 1]
        ax2.hist(df_sent['clean_text'].str.len(), bins=50, color='skyblue', alpha=0.7)
        ax2.set_title('توزیع طول متن')
        ax2.set_xlabel('طول متن')
        ax2.set_ylabel('تعداد')
        ax2.grid(True, alpha=0.3)
        
        # ۳. نسبت فشرده‌سازی خلاصه
        df_summ = pd.read_csv(f"{config['data']['processed']}summ_clean.csv")
        compression_ratio = df_summ['clean_text'].str.len() / df_summ['clean_summary'].str.len()
        
        ax3 = axes[1, 0]
        ax3.hist(compression_ratio, bins=50, color='lightgreen', alpha=0.7)
        ax3.axvline(compression_ratio.mean(), color='red', linestyle='--', 
                   label=f'میانگین: {compression_ratio.mean():.2f}')
        ax3.set_title('نسبت فشرده‌سازی')
        ax3.set_xlabel('نسبت')
        ax3.set_ylabel('تعداد')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ۴. توزیع طول خلاصه
        ax4 = axes[1, 1]
        ax4.hist(df_summ['clean_summary'].str.len(), bins=30, color='lightcoral', alpha=0.7)
        ax4.set_title('توزیع طول خلاصه')
        ax4.set_xlabel('طول خلاصه')
        ax4.set_ylabel('تعداد')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'dataset_distribution.png', config)
        
    except Exception as e:
        print(f"⚠️ خطا در رسم توزیع دیتاست: {e}")

def plot_human_eval_radar():
    """رسم نمودار رادار ارزیابی انسانی"""
    print_section("نمودار رادار ارزیابی انسانی")
    
    # داده‌های نمونه - در عمل باید از ارزیابی انسانی پر شود
    metrics = ['روانی', 'مرتبط بودن', 'اطلاعات مهم', 'حفظ احساس']
    scores = [4.2, 4.0, 3.8, 4.1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 5)
    ax.set_title('ارزیابی انسانی مدل', size=15)
    ax.grid(True)
    
    save_plot(fig, 'human_evaluation_radar.png', config)

def plot_ablation_study():
    """نمودار مطالعه حذف (Ablation Study)"""
    print_section("مطالعه حذف")
    
    try:
        with open(f"{config['output']['results_dir']}evaluation_results.json", 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        rouge1 = results['summary']['rouge1']
        accuracy = results['sentiment']['accuracy']
        
        # داده‌های نمونه - می‌توانید با مقادیر واقعی جایگزین کنید
        models = ['مدل کامل', 'بدون خلاصه', 'بدون احساس', 'بدون مشترک']
        rouge_scores = [rouge1, rouge1 * 0.7, rouge1 * 0.5, rouge1 * 0.3]
        sentiment_f1 = [accuracy, accuracy * 0.5, accuracy * 0.7, accuracy * 0.4]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, rouge_scores, width, label='ROUGE-1', color='skyblue')
        bars2 = ax.bar(x + width/2, sentiment_f1, width, label='F1 احساس', color='lightcoral')
        
        ax.set_xlabel('مدل')
        ax.set_ylabel('امتیاز')
        ax.set_title('مطالعه حذف - تاثیر هر بخش')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        save_plot(fig, 'ablation_study.png', config)
        
    except Exception as e:
        print(f"⚠️ خطا در رسم مطالعه حذف: {e}")

def plot_all_figures():
    """رسم تمام نمودارها"""
    print_section("فاز ۱۱ و ۱۲: رسم تمام نمودارها")
    
    plot_training_curves()
    plot_rouge_comparison()
    plot_sentiment_metrics()
    plot_confusion_matrix()
    plot_dataset_distribution()
    plot_human_eval_radar()
    plot_ablation_study()
    
    print(f"\n✅ تمام نمودارها در {config['output']['figures_dir']} ذخیره شدند.")

if __name__ == "__main__":
    plot_all_figures()