"""
فاز ۴: پیش‌پردازش داده‌ها — نسخه اصلاح‌شده

تغییرات نسبت به نسخه قبلی:
  ۱. PN-Summary: اگر pn_summary.zip موجود باشد، تقسیم رسمی train / dev / test از داخل آن خوانده می‌شود.
     تست رسمی بدون هیچ فیلتر طولی نگه داشته می‌شود تا ROUGE با خط پایه رسمی
     (mT5-base: 45.91 / 27.84 / 39.90) قابل مقایسه باشد.
     فیلترهای طول فقط روی train اعمال می‌شوند.
     در summ_clean.csv ستون split اضافه شده است (train / dev / test).
  ۲. نظرات تکراری دیجی‌کالا حذف می‌شوند.
  ۳. تعداد نمونه‌ها و توزیع کلاس‌ها در پایان چاپ می‌شود.
"""

import re
import pandas as pd
import os
import zipfile
from hazm import Normalizer
from tqdm import tqdm
from utils import load_config, print_section

config = load_config()
normalizer = Normalizer()
SEED = 42
TRAIN_SAMPLE = 20000   # نمونه‌برداری از train برای سرعت (مثل نسخه قبلی)

# ایجاد پوشه‌ها
os.makedirs(config['data']['processed'], exist_ok=True)


def convert_to_3_class(score):
    """تبدیل امتیاز به ۳ کلاس احساسی"""
    if score >= 80:
        return "مثبت"
    elif score >= 50:
        return "خنثی"
    else:
        return "منفی"


def preprocess_text(text):
    """پیش‌پردازش متن (بدون تغییر نسبت به نسخه قبلی)"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = normalizer.normalize(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s؀-ۿﮊپچگ‌]', ' ', text)
    text = re.sub(r'‌', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def read_table(f):
    return pd.read_csv(f, sep='\t', dtype=str, keep_default_na=False)


def load_summary_dataset():
    """
    اولویت ۱: فایل zip رسمی با سه بخش train/dev/test
    اولویت ۲: فایل تکی raw_summ (فقط train) — در این حالت تست رسمی در دسترس نیست
    """
    zip_path = config['data'].get('raw_summ_zip', 'pn_summary.zip')
    parts = []

    if os.path.exists(zip_path):
        print(f"   📦 Read ZIP : {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            for split, keys in {'train': ['train.csv'], 'dev': ['dev.csv', 'valid.csv', 'validation.csv'],
                                'test': ['test.csv']}.items():
                name = next((n for n in names for k in keys if n.lower().endswith(k)), None)
                if name is None:
                    print(f"   ⚠️ بخش {split} در zip پیدا نشد")
                    continue
                with z.open(name) as f:
                    df = read_table(f)
                df['split'] = split
                parts.append(df)
                print(f"   ✅ {split:5}: {len(df):,} Records  ({name})")
    else:
        path = config['data']['raw_summ']
        print(f"   📄 Read File : {path}   (⚠️ zip پیدا نشد؛ تست رسمی استفاده نمی‌شود)")
        with open(path, 'r', encoding='utf-8') as f:
            sep = '\t' if '\t' in f.readline() else ','
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
        df['split'] = 'train'
        parts.append(df)

    df = pd.concat(parts, ignore_index=True)
    result_df = pd.DataFrame({
        'article': df['article'].astype(str),
        'summary': df['summary'].astype(str),
        'split': df['split'],
    })
    result_df = result_df[result_df['article'].str.len() > 10]
    result_df = result_df[result_df['summary'].str.len() > 5]
    return result_df


def main():
    print_section("Phase 4 : Preprocessing")

    # ===== دیتاست احساس =====
    print("\n📝 Preprocessing Sentiment Dataset")
    try:
        df_sent = pd.read_csv(config['data']['raw_sent'])
        print(f"   Primary Count  : {len(df_sent):,}")

        tqdm.pandas(desc="   Text preprocessing")
        df_sent['clean_text'] = df_sent['Text'].astype(str).progress_apply(preprocess_text)
        df_sent = df_sent[df_sent['clean_text'].str.len() > 0]
        df_sent = df_sent[df_sent['clean_text'].str.len() >= config['preprocess']['min_text_len']]
        df_sent = df_sent[df_sent['clean_text'].str.len() <= config['preprocess']['max_text_len']]

        df_sent['Score'] = pd.to_numeric(df_sent['Score'], errors='coerce')
        df_sent = df_sent.dropna(subset=['Score'])
        before = len(df_sent)
        df_sent = df_sent.drop_duplicates(subset='clean_text')
        print(f"   Duplicates removed: {before - len(df_sent):,}")

        df_sent['label'] = df_sent['Score'].apply(convert_to_3_class)
        df_sent = df_sent[['clean_text', 'label', 'Score']]
        df_sent.to_csv(f"{config['data']['processed']}sent_clean.csv", index=False, encoding='utf-8-sig')
        print(f"   ✅ Saved: {len(df_sent):,} samples")
        dist = df_sent['label'].value_counts(normalize=True).mul(100).round(1)
        print("   Class distribution (%): " + " | ".join(f"{k}: {v}" for k, v in dist.items()))

    except FileNotFoundError:
        print(f"❌ فایل {config['data']['raw_sent']} پیدا نشد")
        print("   لطفاً فایل digikala_comments.csv را در مسیر data/raw/ بگذارید")
        return

    # ===== دیتاست خلاصه‌سازی =====
    print("\n📝 Preprocessing Summarization Dataset...")
    try:
        df_summ = load_summary_dataset()

        # نمونه‌برداری فقط از train
        train = df_summ[df_summ['split'] == 'train']
        if len(train) > TRAIN_SAMPLE:
            train = train.sample(n=TRAIN_SAMPLE, random_state=SEED)
            print(f"   Train sampled: {len(train):,}")
        df_summ = pd.concat([train, df_summ[df_summ['split'] != 'train']], ignore_index=True)

        tqdm.pandas(desc="   Preprocessing data")
        df_summ['clean_text'] = df_summ['article'].astype(str).progress_apply(preprocess_text)
        df_summ['clean_summary'] = df_summ['summary'].astype(str).progress_apply(preprocess_text)

        # ردیف‌های خالی از همه بخش‌ها حذف می‌شوند
        df_summ = df_summ[(df_summ['clean_text'].str.len() > 0) & (df_summ['clean_summary'].str.len() > 0)]

        # فیلترهای طول فقط روی train (تست و dev رسمی دست‌نخورده می‌مانند)
        is_tr = df_summ['split'] == 'train'
        keep = (~is_tr) | (
            (df_summ['clean_text'].str.len() >= config['preprocess']['min_text_len']) &
            (df_summ['clean_text'].str.len() <= config['preprocess']['max_text_len']) &
            (df_summ['clean_summary'].str.len() >= 10) &
            (df_summ['clean_summary'].str.len() <= 256)
        )
        df_summ = df_summ[keep]

        df_summ = df_summ[['clean_text', 'clean_summary', 'split']]
        df_summ.to_csv(f"{config['data']['processed']}summ_clean.csv", index=False, encoding='utf-8-sig')
        counts = df_summ['split'].value_counts()
        print(f"   ✅ Saved: {len(df_summ):,} samples  ->  " +
              " | ".join(f"{s}: {counts.get(s, 0):,}" for s in ['train', 'dev', 'test']))

    except Exception as e:
        print(f"❌ Error in loading summarization dataset: {e}")
        return

    print("\n✅ Processing has been completed!")


if __name__ == "__main__":
    main()
