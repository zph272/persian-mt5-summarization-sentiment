"""
فاز ۲: تحلیل و بررسی دیتاست‌ها (نسخه نهایی - کاملاً بدون خطا)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import load_config, print_section, save_plot, save_json, fa

config = load_config()

# ایجاد پوشه‌ها
os.makedirs(config['output']['figures_dir'], exist_ok=True)
os.makedirs(config['output']['results_dir'], exist_ok=True)

# ترتیب ثابت کلاس‌ها تا رنگ هر کلاس در نمودار همیشه درست باشد
SENTIMENT_CLASSES = ["منفی", "خنثی", "مثبت"]
CLASS_EN = {"منفی": "Negative", "خنثی": "Neutral", "مثبت": "Positive"}  # نام انگلیسی کلاس‌ها برای چاپ در ترمینال
SENTIMENT_COLORS = ['#E74C3C', '#F1C40F', '#2ECC71']

# نام‌های احتمالی ستون‌ها به ترتیب اولویت
TEXT_COL_CANDIDATES = ['article', 'text', 'content', 'document']
SUMMARY_COL_CANDIDATES = ['summary', 'abstract', 'title']

def convert_to_3_class(score):
    """تبدیل امتیاز به ۳ کلاس احساسی"""
    if score >= 80:
        return "مثبت"
    elif score >= 50:
        return "خنثی"
    else:
        return "منفی"

def length_stats(series):
    """آمار طول: کاراکتر و کلمه (برای جدول ۲ مقاله، داور طول را بر حسب توکن/کلمه خواسته است)"""
    s = series.astype(str)
    chars = s.str.len()
    words = s.str.split().str.len()
    return {
        'chars': {'mean': float(chars.mean()), 'median': float(chars.median()),
                  'max': int(chars.max()), 'min': int(chars.min())},
        'words': {'mean': float(words.mean()), 'median': float(words.median()),
                  'p95': float(words.quantile(0.95)), 'max': int(words.max()), 'min': int(words.min()),
                  'pct_over_100': float((words > 100).mean() * 100)},
    }


def print_length_stats(title, st):
    c, w = st['chars'], st['words']
    print(f"\n📏 {title}:")
    print(f"   Characters → Mean: {c['mean']:.1f} | Median: {c['median']:.0f} | Min: {c['min']:,} | Max: {c['max']:,}")
    print(f"   Words      → Mean: {w['mean']:.1f} | Median: {w['median']:.0f} | 95th pct: {w['p95']:.0f} "
          f"| Max: {w['max']:,} | >100 words: {w['pct_over_100']:.1f}%")


def find_column(columns, candidates, exclude=None):
    """پیدا کردن ستون: اول تطابق دقیق به ترتیب اولویت، بعد تطابق جزئی"""
    columns = [c for c in columns if c != exclude]
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for cand in candidates:
        for col in columns:
            if cand in col.lower():
                return col
    return None



def load_summary_dataset_from_zip():
    """بارگذاری دیتاست خلاصه‌سازی از فایل محلی (مثل دیجی‌کالا)"""
    path = config['data']['raw_summ']
    print(f"   📄 Reading file: {path}")

    # تشخیص جداکننده: فایل‌های pn_summary معمولاً با Tab جدا شده‌اند
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    sep = '\t' if '\t' in first_line else ','
    df = pd.read_csv(path, sep=sep)

    print(f"   ✅ {len(df):,} records | columns: {list(df.columns)}")

    result_df = pd.DataFrame({
        'article': df['article'].astype(str),
        'summary': df['summary'].astype(str)
    })
    result_df = result_df[result_df['article'].str.len() > 10]
    result_df = result_df[result_df['summary'].str.len() > 5]

    print(f"   ✅ After cleaning: {len(result_df):,} records")
    return result_df

def analyze_sentiment_dataset(df):
    """تحلیل دیتاست احساس"""
    print_section("Sentiment Dataset Analysis (Digikala Comments)")

    # حذف ردیف‌هایی که امتیاز یا متن ندارند
    df = df.copy()
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['Score', 'Text'])
    dropped = before - len(df)
    if dropped:
        print(f"🧹 {dropped:,} rows removed (missing score or text).")

    # تبدیل به ۳ کلاس
    df['sentiment'] = df['Score'].apply(convert_to_3_class)

    total_samples = len(df)
    print(f"📊 Total samples: {total_samples:,}")

    # توزیع کلاس‌ها (با ترتیب ثابت؛ کلاس بدون نمونه مقدار صفر می‌گیرد)
    class_counts = df['sentiment'].value_counts().reindex(SENTIMENT_CLASSES, fill_value=0)
    print(f"\n📈 Sentiment class distribution (3 classes):")
    for label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"   {CLASS_EN.get(label, label)}: {count:,} ({percentage:.2f}%)")

    # آمار طول متن
    text_lengths = df['Text'].astype(str).str.len()
    text_stats = length_stats(df['Text'])
    print_length_stats("Review length", text_stats)

    # نظرات تکراری (روی اندازه واقعی داده اثر دارد)
    n_dup = int(df['Text'].astype(str).str.strip().duplicated().sum())
    print(f"\n🔁 Duplicate reviews: {n_dup:,}")

    # نسبت نامتوازنی (اگر کلاسی خالی باشد، بی‌نهایت است)
    if class_counts.min() > 0:
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\n⚖️ Imbalance ratio: {imbalance_ratio:.2f}")
    else:
        imbalance_ratio = None
        print(f"\n⚠️ At least one class has no samples; imbalance ratio cannot be computed.")

    # رسم نمودار
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # نمودار توزیع کلاس‌ها
        bars = axes[0].bar([fa(c) for c in class_counts.index], class_counts.values,
                           color=SENTIMENT_COLORS, edgecolor='white', width=0.6)
        # نوشتن تعداد و درصد بالای هر میله
        for bar, count in zip(bars, class_counts.values):
            axes[0].annotate(f'{count:,}\n({count / total_samples * 100:.1f}%)',
                             (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha='center', va='bottom', fontsize=11, xytext=(0, 4),
                             textcoords='offset points')
        axes[0].set_ylim(0, class_counts.max() * 1.18)
        axes[0].set_title(fa('توزیع کلاس‌های احساسی'), fontweight='bold')
        axes[0].set_xlabel(fa('کلاس'))
        axes[0].set_ylabel(fa('تعداد'))
        axes[0].grid(axis='x', visible=False)

        # نمودار توزیع طول متن
        # محور افقی تا صدک ۹۹ تا چند متن خیلی بلند نمودار را فشرده نکنند
        x_max = text_lengths.quantile(0.99)
        axes[1].hist(text_lengths[text_lengths <= x_max], bins=40, color='#5DADE2', edgecolor='white')
        axes[1].set_xlim(0, x_max)
        axes[1].axvline(text_lengths.mean(), color='#C0392B', linestyle='--', linewidth=1.5,
                       label=fa(f'میانگین: {text_lengths.mean():.0f}'))
        axes[1].set_title(fa('توزیع طول متن'), fontweight='bold')
        axes[1].set_xlabel(fa('طول متن'))
        axes[1].set_ylabel(fa('تعداد'))
        axes[1].legend()

        plt.tight_layout()
        save_plot(fig, 'sentiment_analysis.png', config)

    except Exception as e:
        print(f"⚠️ Plotting error: {e}")

    # تبدیل به JSON-friendly
    return {
        'total_samples': int(total_samples),
        'class_distribution': {str(k): int(v) for k, v in class_counts.items()},
        'duplicates': n_dup,
        'text_length_stats': text_stats,
        'imbalance_ratio': float(imbalance_ratio) if imbalance_ratio is not None else None
    }

def analyze_summarization_dataset(df):
    """تحلیل دیتاست خلاصه‌سازی"""
    print_section("Summarization Dataset Analysis (PN Summary)")

    total_samples = len(df)
    print(f"📊 Total samples: {total_samples:,}")

    text_lengths = df['article'].astype(str).str.len()
    summary_lengths = df['summary'].astype(str).str.len()

    article_stats = length_stats(df['article'])
    summary_stats = length_stats(df['summary'])
    print_length_stats("Article length", article_stats)
    print_length_stats("Summary length", summary_stats)

    # نسبت فشرده‌سازی (بر حسب کاراکتر و کلمه)
    compression_ratio = text_lengths.mean() / summary_lengths.mean()
    compression_words = article_stats['words']['mean'] / summary_stats['words']['mean']
    print(f"\n📦 Compression ratio: {compression_ratio:.2f} (chars) | {compression_words:.2f} (words)")

    # رسم نمودار
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x_max = text_lengths.quantile(0.99)
        axes[0].hist(text_lengths[text_lengths <= x_max], bins=40, color='#5DADE2', edgecolor='white')
        axes[0].set_xlim(0, x_max)
        axes[0].axvline(text_lengths.mean(), color='#C0392B', linestyle='--', linewidth=1.5,
                       label=fa(f'میانگین: {text_lengths.mean():.0f}'))
        axes[0].set_title(fa('توزیع طول متن اصلی'), fontweight='bold')
        axes[0].set_xlabel(fa('طول متن'))
        axes[0].set_ylabel(fa('تعداد'))
        axes[0].legend()

        s_max = summary_lengths.quantile(0.99)
        axes[1].hist(summary_lengths[summary_lengths <= s_max], bins=40, color='#EC7063', edgecolor='white')
        axes[1].set_xlim(0, s_max)
        axes[1].axvline(summary_lengths.mean(), color='#C0392B', linestyle='--', linewidth=1.5,
                       label=fa(f'میانگین: {summary_lengths.mean():.0f}'))
        axes[1].set_title(fa('توزیع طول خلاصه'), fontweight='bold')
        axes[1].set_xlabel(fa('طول خلاصه'))
        axes[1].set_ylabel(fa('تعداد'))
        axes[1].legend()

        plt.tight_layout()
        save_plot(fig, 'summary_analysis.png', config)

    except Exception as e:
        print(f"⚠️ Plotting error: {e}")

    # تبدیل به JSON-friendly
    return {
        'total_samples': int(total_samples),
        'text_length_stats': article_stats,
        'summary_length_stats': summary_stats,
        'compression_ratio': float(compression_ratio),
        'compression_ratio_words': float(compression_words)
    }

def main():
    print_section("Phase 2: Dataset Analysis")

    # ===== دیتاست احساس =====
    try:
        df_sent = pd.read_csv(config['data']['raw_sent'])
        sent_stats = analyze_sentiment_dataset(df_sent)
    except FileNotFoundError:
        print(f"❌ File not found: {config['data']['raw_sent']}")
        print("   Please place digikala_comments.csv in data/raw/")
        return

    # ===== دیتاست خلاصه‌سازی =====
    try:
        df_summ = load_summary_dataset_from_zip()
        summ_stats = analyze_summarization_dataset(df_summ)
    except Exception as e:
        print(f"❌ Error loading summarization dataset: {e}")
        return

    # ذخیره نتایج
    all_stats = {
        'sentiment': sent_stats,
        'summarization': summ_stats,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    results_path = os.path.join(config['output']['results_dir'], 'dataset_analysis.json')
    save_json(all_stats, results_path)

    print(f"\n✅ Analysis complete! Results saved to {results_path}")

if __name__ == "__main__":
    main()
