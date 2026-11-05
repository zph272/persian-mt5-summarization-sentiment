"""
ماژول کمکی پروژه (utils.py)
این فایل مستقیم اجرا نمی‌شود؛ بقیه‌ی فایل‌ها توابع آن را import می‌کنند.
"""

# ===== وارد کردن کتابخانه‌ها =====
import yaml                       # خواندن فایل تنظیمات config.yaml
import json                       # ذخیره و خواندن فایل‌های JSON
from pathlib import Path          # کار راحت‌تر با مسیر فایل‌ها
import pandas as pd               # کار با جدول داده‌ها
import numpy as np                # محاسبات عددی
import matplotlib.pyplot as plt   # رسم نمودار
import seaborn as sns             # نمودارهای آماری زیباتر
from datetime import datetime     # تاریخ و ساعت
import os                         # ساختن پوشه‌ها و کار با سیستم‌عامل
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # hide 'font not found' warnings

# ===== تنظیم فونت و ظاهر نمودارها =====
# فونت‌هایی که حروف فارسی را پشتیبانی می‌کنند (به ترتیب اولویت)
plt.rcParams['font.family'] = ['Vazirmatn', 'Tahoma', 'DejaVu Sans']
# نمایش درست علامت منفی روی محورها
plt.rcParams['axes.unicode_minus'] = False
# اندازه‌ی فونت‌ها (مناسب چاپ در پایان‌نامه)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 13
# حذف قاب بالا و راست نمودار برای ظاهر تمیزتر
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# خطوط شبکه‌ی کم‌رنگ و پشت نمودار
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.axisbelow'] = True


def fa(text):
    """متن فارسی نمودارها

    نسخه‌ی matplotlib این سیستم خودش حروف فارسی را می‌چسباند و راست‌به‌چپ می‌کند
    (تست font_test.py، گزینه‌ی ۱)؛ پس متن بدون تغییر برگردانده می‌شود.
    اگر روی سیستم دیگری فارسی به‌هم‌ریخته بود، فقط همین تابع را تغییر دهید.
    """
    return str(text)


SMOKE_ROWS = 24   # rows per split in the smoke test


def load_config():
    """بارگذاری فایل کانفیگ

    حالت آزمایشی (SMOKE=1): فقط برای اطمینان از اجرا شدن کد، بدون GPU.
    مدل کوچک mt5-small، ۱ ایپاک، ۲۴ نمونه، و خروجی در پوشه‌ی جداگانه smoke_outputs/
    (با خروجی اصلی قاطی نمی‌شود). نتایج این حالت معنی ندارند.
    """
    # فایل config.yaml از پوشه‌ای خوانده می‌شود که برنامه در آن اجرا شده است
    with open('config.yaml', 'r', encoding='utf-8') as f:
        # تبدیل محتوای فایل به دیکشنری پایتون
        cfg = yaml.safe_load(f)
    if os.environ.get('SMOKE') == '1':
        cfg['model']['base_model'] = 'google/mt5-small'
        for k in ('epochs_stage1', 'epochs_stage2', 'epochs_joint'):
            cfg['model'][k] = 1
        cfg['model']['warmup_steps'] = 0
        cfg['output'] = {k: v.replace('outputs/', 'smoke_outputs/', 1) for k, v in cfg['output'].items()}
        cfg['output']['checkpoints_dir'] = '/tmp/smoke_checkpoints/'
        cfg['smoke_rows'] = SMOKE_ROWS
    return cfg


def create_directories():
    """ایجاد پوشه‌های مورد نیاز"""
    config = load_config()
    # فهرست پوشه‌هایی که پروژه لازم دارد (مسیرها از config.yaml می‌آیند)
    dirs = [
        config['data']['processed'],          # داده‌های تمیزشده
        config['output']['models_dir'],       # مدل‌های آموزش‌دیده
        config['output']['results_dir'],      # نتایج عددی
        config['output']['figures_dir'],      # نمودارها
        config['output']['checkpoints_dir']   # نسخه‌های میانی مدل
    ]
    # ساختن هر پوشه؛ اگر از قبل وجود داشته باشد خطا نمی‌دهد
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(data, path):
    """ذخیره دیتا به صورت JSON"""
    # اگر پوشه‌ی مقصد وجود ندارد، آن را بساز
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False: حروف فارسی به همان شکل ذخیره شوند، نه به صورت کد
        # indent=2: فایل مرتب و خوانا باشد
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    """بارگذاری فایل JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_plot(fig, name, config):
    """ذخیره نمودار با تنظیمات مناسب"""
    # اگر پوشه‌ی نمودارها وجود ندارد، آن را بساز
    os.makedirs(config['output']['figures_dir'], exist_ok=True)
    # مسیر کامل فایل = پوشه‌ی نمودارها + نام فایل
    path = Path(config['output']['figures_dir']) / name
    # ذخیره با کیفیت بالا (۳۰۰ DPI) و بدون حاشیه‌ی اضافه
    fig.savefig(path, dpi=300, bbox_inches='tight')
    # بستن نمودار برای آزاد شدن حافظه
    plt.close(fig)
    print(f"Plot saved: {path}")


def get_timestamp():
    """دریافت زمان فعلی برای نام‌گذاری"""
    # مثال خروجی: 20261029_134800 (سال‌ماه‌روز_ساعت‌دقیقه‌ثانیه)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title):
    """چاپ عنوان بخش"""
    # چاپ عنوان بین دو خط جداکننده در ترمینال
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60 + "\n")
