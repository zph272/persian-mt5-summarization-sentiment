"""
Phase 11 - Figures and results table from REAL outputs only (fixed version)

  python 8_plot_results.py

Reads outputs/results/*_training_history.csv and *_eval.json and writes:
  outputs/figures/loss_curves.png           training / validation loss vs. optimization step (Fig. 3)
  outputs/figures/confusion_<model>.png     confusion matrix (Fig. 2)
  outputs/results/results_table.md          all models and baselines in one table

Changes vs. previous version:
  - Removed the "human evaluation" radar chart and the "ablation" chart: they used hard-coded
    numbers (4.2, 4.0, ...; accuracy x 0.5, x 0.7, ...) that were not measured.
  - Validation loss is plotted at its real optimization step (previously spread evenly along the
    training curve, which made the two curves look misaligned).
  - English labels for publication.
"""
import glob
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_config

config = load_config()
RES = config['output']['results_dir']
FIG = config['output']['figures_dir']
os.makedirs(FIG, exist_ok=True)

EN = {"مثبت": "Positive", "خنثی": "Neutral", "منفی": "Negative", "نامعتبر": "Invalid"}
NAMES = {
    'summ': 'Stage 1 (summarization)', 'summ_final': 'Stage 1 (summarization)',
    'sent_final': 'Stages 1→2', 'joint_final': 'Stages 1→2→3 (final model)',
    'sent_only': 'Ablation: sentiment only (from mT5-base)',
    'joint_only': 'Ablation: joint only, no curriculum (from mT5-base)',
}
ORDER = ['summ_final', 'sent_final', 'joint_final', 'sent_only', 'joint_only']


def plot_losses():
    files = sorted(glob.glob(f"{RES}*_training_history.csv"))
    if not files:
        print("⚠️ No training-history files found.")
        return
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 3.8), squeeze=False)
    for ax, f in zip(axes[0], files):
        key = os.path.basename(f).replace('_training_history.csv', '')
        h = pd.read_csv(f)
        if 'loss' in h:
            t = h.dropna(subset=['loss'])
            ax.plot(t['step'], t['loss'], label='Training loss (as logged)', alpha=0.85)
        if 'eval_loss' in h:
            e = h.dropna(subset=['eval_loss'])
            ax.plot(e['step'], e['eval_loss'], 'o-', ms=3, label='Validation loss')
        ax.set_title(NAMES.get(key, key), fontsize=10)
        ax.set_xlabel('Optimization step')
        ax.set_ylabel('Cross-entropy')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}loss_curves.png", dpi=300)
    plt.close(fig)
    print(f"✅ {FIG}loss_curves.png")


def plot_confusions():
    for f in sorted(glob.glob(f"{RES}*_eval.json")):
        r = json.load(open(f, encoding='utf-8'))
        if 'sentiment' not in r or 'model' not in r:
            continue
        cm = r['sentiment']['confusion_matrix']
        vals, cols = np.array(cm['values']), cm['cols_pred']
        if vals[:, -1].sum() == 0:                      # hide the "invalid" column if empty
            vals, cols = vals[:, :-1], cols[:-1]
        fig, ax = plt.subplots(figsize=(4.8, 3.9))
        im = ax.imshow(vals, cmap='Blues')
        ax.set_xticks(range(len(cols)), [EN[c] for c in cols])
        ax.set_yticks(range(len(cm['rows_true'])), [EN[c] for c in cm['rows_true']])
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, int(vals[i, j]), ha='center', va='center',
                        color='white' if vals[i, j] > vals.max() / 2 else 'black')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(f"{FIG}confusion_{r['model']}.png", dpi=300)
        plt.close(fig)
        print(f"✅ {FIG}confusion_{r['model']}.png")


def fmt(x, pct=False):
    if x is None:
        return '–'
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def results_table():
    rows = []
    bpath = f"{RES}baselines_eval.json"
    if os.path.exists(bpath):
        b = json.load(open(bpath, encoding='utf-8'))
        ms = b['majority_sentiment']
        rows.append([f"Majority class ({EN[ms['majority_label']]})", None, None, None, ms['accuracy'],
                     ms['macro_f1'], ms['weighted_f1'], ms['per_class']['منفی']['recall'], 0])
        lk = [k for k in b if k.startswith('lead')][0]
        rows.append([f"Lead baseline ({lk[4:-8]} words)", b[lk]['rouge1'], b[lk]['rouge2'], b[lk]['rougeL']] + [None] * 5)
    rows.append(["mT5-base, full training set (published)", 45.91, 27.84, 39.90] + [None] * 5)
    for name in ORDER:
        f = f"{RES}{name}_eval.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f, encoding='utf-8'))
        s, m = r.get('summarization', {}), r.get('sentiment', {})
        rows.append([NAMES.get(name, name), s.get('rouge1'), s.get('rouge2'), s.get('rougeL'),
                     m.get('accuracy'), m.get('macro_f1'), m.get('weighted_f1'),
                     m.get('per_class', {}).get('منفی', {}).get('recall'), m.get('invalid_outputs')])
    lines = ["| Model | R-1 | R-2 | R-L | Acc (%) | Macro-F1 (%) | Weighted-F1 (%) | Neg. recall (%) | Invalid |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {fmt(r[1])} | {fmt(r[2])} | {fmt(r[3])} | {fmt(r[4], True)} | "
                     f"{fmt(r[5], True)} | {fmt(r[6], True)} | {fmt(r[7], True)} | {'–' if r[8] is None else r[8]} |")
    md = "\n".join(lines)
    with open(f"{RES}results_table.md", 'w', encoding='utf-8') as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    plot_losses()
    plot_confusions()
    results_table()
