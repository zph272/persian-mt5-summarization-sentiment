"""
فاز ۳: آماده‌سازی داده‌های مشترک — نسخه اصلاح‌شده

تغییرات نسبت به نسخه قبلی:
  ۱. جفت‌سازی تصادفی «نظر دیجی‌کالا + خلاصه خبر بی‌ربط» حذف شد.
     هر نمونه با پیشوند وظیفه‌اش وارد مدل می‌شود:
        «خلاصه کن: <متن خبر>»   →  <خلاصه واقعی همان خبر>
        «احساس: <نظر دیجی‌کالا>» →  مثبت / خنثی / منفی
  ۲. متن خبرهای PN-Summary هم استفاده می‌شود (قبلاً دور ریخته می‌شد).
  ۳. همه نمونه‌گیری‌ها seed دارند ← هر بار اجرا همان نتیجه را می‌دهد.
  ۴. هر دیتاست جداگانه به train/val/test تقسیم می‌شود (احساس به‌صورت طبقه‌بندی‌شده).
  ۵. joint_clean.csv فقط شامل داده train است تا اسکریپت مرحله ۳ هیچ نمونه تستی نبیند.

ورودی‌ها:  sent_clean.csv (clean_text, label) , summ_clean.csv (clean_text, clean_summary)
خروجی‌ها: joint_train.csv, joint_val.csv, joint_test.csv, joint_clean.csv
           ستون‌ها: text, target, task   (task = summ یا sent)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils import load_config, print_section

config = load_config()
SEED = 42
MAX_PER_TASK = 10000          # حداکثر نمونه از هر دیتاست (مثل نسخه قبلی)
PREFIX_SUMM = "خلاصه کن: "
PREFIX_SENT = "احساس: "

out_dir = config['data']['processed']
os.makedirs(out_dir, exist_ok=True)


def split_70_15_15(df, stratify_col=None):
    """تقسیم ۷۰/۱۵/۱۵ با seed ثابت."""
    strat = df[stratify_col] if stratify_col else None
    train, temp = train_test_split(df, test_size=0.3, random_state=SEED, stratify=strat)
    strat = temp[stratify_col] if stratify_col else None
    val, test = train_test_split(temp, test_size=0.5, random_state=SEED, stratify=strat)
    return train, val, test


def create_joint_dataset(df_sent, df_summ):
    print_section("فاز ۳: آماده‌سازی داده‌های مشترک (اصلاح‌شده)")

    # پاک‌سازی ردیف‌های خالی و تکراری
    df_sent = df_sent.dropna(subset=['clean_text', 'label']).drop_duplicates(subset='clean_text')
    df_summ = df_summ.dropna(subset=['clean_text', 'clean_summary'])
    if 'split' in df_summ.columns:
        # تکراری‌ها داخل هر بخش حذف می‌شوند؛ هر متن train/dev که در تست هم باشد از train/dev حذف می‌شود (تست دست‌نخورده)
        df_summ = df_summ.drop_duplicates(subset=['clean_text', 'split'])
        test_texts = set(df_summ.loc[df_summ['split'] == 'test', 'clean_text'])
        df_summ = df_summ[(df_summ['split'] == 'test') | (~df_summ['clean_text'].isin(test_texts))]
    else:
        df_summ = df_summ.drop_duplicates(subset='clean_text')

    # نمونه‌برداری با seed (احساس: طبقه‌بندی‌شده تا نسبت کلاس‌ها حفظ شود)
    if len(df_sent) > MAX_PER_TASK:
        df_sent, _ = train_test_split(df_sent, train_size=MAX_PER_TASK,
                                      stratify=df_sent['label'], random_state=SEED)
    def to_summ(df):
        return pd.DataFrame({
            'text': PREFIX_SUMM + df['clean_text'].astype(str),
            'target': df['clean_summary'].astype(str),
            'task': 'summ',
        })

    # ساخت نمونه‌های هر وظیفه — بدون هیچ جفت‌سازی
    if 'split' in df_summ.columns and (df_summ['split'] == 'test').any():
        # تقسیم رسمی PN-Summary (از 2_preprocess.py)؛ تست رسمی کامل و بدون نمونه‌برداری
        print("📌 PN-Summary: استفاده از تقسیم رسمی train / dev / test")
        tr = df_summ[df_summ['split'] == 'train']
        va = df_summ[df_summ['split'] == 'dev']
        te = df_summ[df_summ['split'] == 'test']
        n_tr, n_va = int(MAX_PER_TASK * 0.7), int(MAX_PER_TASK * 0.15)
        tr = tr.sample(n=min(n_tr, len(tr)), random_state=SEED)
        va = va.sample(n=min(n_va, len(va)), random_state=SEED)
        s_tr, s_va, s_te = to_summ(tr), to_summ(va), to_summ(te)
        summ = pd.concat([s_tr, s_va, s_te])
    else:
        print("⚠️ PN-Summary: تقسیم رسمی پیدا نشد؛ تقسیم ۷۰/۱۵/۱۵ از داده موجود")
        if len(df_summ) > MAX_PER_TASK:
            df_summ = df_summ.sample(n=MAX_PER_TASK, random_state=SEED)
        summ = to_summ(df_summ)
        s_tr, s_va, s_te = split_70_15_15(summ)

    sent = pd.DataFrame({
        'text': PREFIX_SENT + df_sent['clean_text'].astype(str),
        'target': df_sent['label'].astype(str),
        'task': 'sent',
    })
    t_tr, t_va, t_te = split_70_15_15(sent, stratify_col='target')

    splits = {}
    for name, (a, b) in {'train': (s_tr, t_tr), 'val': (s_va, t_va), 'test': (s_te, t_te)}.items():
        df = pd.concat([a, b], ignore_index=True)
        df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # مخلوط کردن دو وظیفه
        df.to_csv(f"{out_dir}joint_{name}.csv", index=False)
        splits[name] = df

    # joint_clean.csv = فقط train ← اسکریپت مرحله ۳ از این فایل می‌خواند و نباید تست را ببیند
    splits['train'].to_csv(f"{out_dir}joint_clean.csv", index=False)

    # بررسی نشت داده
    overlap = len(set(splits['train']['text']) & set(splits['test']['text']))

    print(f"📊 خلاصه‌سازی: {len(summ):,} نمونه   |   احساس: {len(sent):,} نمونه")
    for name, df in splits.items():
        c = df['task'].value_counts()
        print(f"   {name:5}: {len(df):,}  (خلاصه: {c.get('summ', 0):,}  احساس: {c.get('sent', 0):,})")
    print("\n📊 توزیع کلاس احساس در تست:")
    print(t_te['target'].value_counts(normalize=True).mul(100).round(1).to_string())
    print(f"\n🔒 هم‌پوشانی train و test: {overlap}  (باید صفر باشد)")
    print(f"\n📝 نمونه خلاصه‌سازی:\n   ورودی: {s_tr.iloc[0]['text'][:100]}...\n   هدف:  {s_tr.iloc[0]['target'][:100]}")
    print(f"📝 نمونه احساس:\n   ورودی: {t_tr.iloc[0]['text'][:100]}...\n   هدف:  {t_tr.iloc[0]['target']}")
    print(f"\n✅ ذخیره شد: joint_train / joint_val / joint_test / joint_clean در {out_dir}")
    return splits


def main():
    df_sent = pd.read_csv(f"{out_dir}sent_clean.csv")
    df_summ = pd.read_csv(f"{out_dir}summ_clean.csv")
    print(f"📂 دیتاست احساس: {len(df_sent):,} نمونه")
    print(f"📂 دیتاست خلاصه‌سازی: {len(df_summ):,} نمونه")
    create_joint_dataset(df_sent, df_summ)


if __name__ == "__main__":
    main()
