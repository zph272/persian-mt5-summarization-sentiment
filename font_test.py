"""
تست نمایش فارسی در نمودار — سه روش را کنار هم رسم می‌کند تا ببینیم کدام درست است.
اجرا:  & D:\my-p\python.exe font_test.py
"""
import matplotlib
import matplotlib.pyplot as plt

print("matplotlib version:", matplotlib.__version__)

plt.rcParams['font.family'] = ['Vazirmatn', 'Tahoma', 'DejaVu Sans']

sample = "توزیع کلاس‌های احساسی - میانگین: 248"

versions = [("1: raw text", sample)]
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    versions.append(("2: reshape only", arabic_reshaper.reshape(sample)))
    versions.append(("3: reshape + bidi", get_display(arabic_reshaper.reshape(sample))))
except ImportError:
    print("arabic-reshaper / python-bidi not installed")

fig, axes = plt.subplots(len(versions), 1, figsize=(8, 1.4 * len(versions)))
if len(versions) == 1:
    axes = [axes]
for ax, (name, txt) in zip(axes, versions):
    ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=18)
    ax.set_title(name, fontsize=10, loc='left')
    ax.axis('off')

plt.tight_layout()
fig.savefig("font_test.png", dpi=150)
print("Saved: font_test.png")
