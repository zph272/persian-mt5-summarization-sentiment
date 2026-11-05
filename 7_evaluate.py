"""
Phase 9 - Evaluation on the held-out TEST split (fixed version)

  python 7_evaluate.py --model joint_final                 # final model, both tasks
  python 7_evaluate.py --model summ_final --tasks summ     # Stage 1 model, summarization only
  python 7_evaluate.py --model sent_only  --tasks sent     # ablation model, sentiment only
  python 7_evaluate.py --baselines                         # majority-class and Lead baselines (no GPU needed)
  python 7_evaluate.py --model joint_final --tasks loss    # train vs. validation loss on the same scale

Changes vs. previous version:
  1. Summarization and sentiment are evaluated separately, each with its own task prefix:
     5,593 official PN-Summary test articles and 449 held-out Digikala reviews (previously head(2000)).
  2. Fixed regex: outputs that are not exactly one label count as errors ("invalid"), not "neutral".
  3. ROUGE with whitespace tokenization (the default rouge_score tokenizer deletes Persian letters).
  4. Macro-F1, per-class results and negative-class recall are reported.
  5. Batched generation (much faster) and beam search with 4 beams.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import MT5ForConditionalGeneration, AutoTokenizer

from utils import print_section
from train_common import (config, load_split, model_path, parse_sentiment, LABELS, INVALID,
                          RESULTS_DIR, print_gpu)

NUM_BEAMS = 4
LEAD_WORDS = 27        # Lead baseline length = median PN-Summary summary length (words)


class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()


def rouge(preds, refs):
    sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=WhitespaceTokenizer())
    s = [sc.score(r, p) for p, r in zip(preds, refs)]          # score(target, prediction)
    return {k: round(100 * float(np.mean([x[k].fmeasure for x in s])), 2) for k in ('rouge1', 'rouge2', 'rougeL')}


def sentiment_report(refs, preds):
    p, r, f, s = precision_recall_fscore_support(refs, preds, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(refs, preds, labels=LABELS, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(refs, preds, labels=LABELS, average='weighted', zero_division=0)
    cm = confusion_matrix(refs, preds, labels=LABELS + [INVALID])[:len(LABELS)]
    return {
        'n': len(refs), 'accuracy': float(accuracy_score(refs, preds)),
        'macro_precision': float(mp), 'macro_recall': float(mr), 'macro_f1': float(mf),
        'weighted_precision': float(wp), 'weighted_recall': float(wr), 'weighted_f1': float(wf),
        'per_class': {lab: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(s[i])}
                      for i, lab in enumerate(LABELS)},
        'invalid_outputs': int(sum(x == INVALID for x in preds)),
        'confusion_matrix': {'rows_true': LABELS, 'cols_pred': LABELS + [INVALID], 'values': cm.tolist()},
    }


def generate(model, tokenizer, texts, max_new, beams, batch_size, desc):
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))   # similar lengths together = faster
    out = [None] * len(texts)
    for s in tqdm(range(0, len(order), batch_size), desc=desc):
        idx = order[s:s + batch_size]
        enc = tokenizer([texts[i] for i in idx], max_length=config['data']['max_input_length'],
                        truncation=True, padding=True, return_tensors='pt').to(model.device)
        kw = dict(max_new_tokens=max_new, num_beams=beams)
        if beams > 1:
            kw.update(early_stopping=True, no_repeat_ngram_size=3)
        with torch.no_grad():
            g = model.generate(**enc, **kw)
        for i, t in zip(idx, tokenizer.batch_decode(g, skip_special_tokens=True)):
            out[i] = t.strip()
    return out


def mean_loss(model, tokenizer, df, max_target_len, n=500, batch_size=8):
    """Average per-batch cross-entropy in eval mode (no dropout) on up to n samples."""
    n = config.get('smoke_rows', n)
    df = df.sample(n=min(n, len(df)), random_state=42)
    losses = []
    model.eval()
    for s in range(0, len(df), batch_size):
        b = df.iloc[s:s + batch_size]
        enc = tokenizer(list(b['text']), max_length=config['data']['max_input_length'], truncation=True,
                        padding=True, return_tensors='pt').to(model.device)
        lab = tokenizer(text_target=list(b['target']), max_length=max_target_len, truncation=True,
                        padding=True, return_tensors='pt')['input_ids']
        lab[lab == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            losses.append(model(**enc, labels=lab.to(model.device)).loss.item())
    return float(np.mean(losses))


def save(name, results):
    path = f"{RESULTS_DIR}{name}_eval.json"
    old = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    old.update(results)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {path}")


def run_baselines():
    print_section("Baselines (no model)")
    train_sent = load_split('train', 'sent')
    test_sent = load_split('test', 'sent')
    majority = train_sent['target'].value_counts().idxmax()
    sent = sentiment_report(list(test_sent['target']), [majority] * len(test_sent))
    sent['majority_label'] = majority
    print(f"   Majority class ({majority}): accuracy={sent['accuracy']:.3f}  macro-F1={sent['macro_f1']:.3f}")

    test_summ = load_split('test', 'summ')
    articles = [t.replace("خلاصه کن: ", "", 1) for t in test_summ['text']]
    lead = [" ".join(a.split()[:LEAD_WORDS]) for a in articles]
    r = rouge(lead, list(test_summ['target']))
    print(f"   Lead-{LEAD_WORDS} words: {r}")
    save('baselines', {'majority_sentiment': sent, f'lead{LEAD_WORDS}_summary': r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='joint_final', help='model name in outputs/models/')
    ap.add_argument('--tasks', default='summ,sent', help='any of: summ, sent, loss')
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--max_summ', type=int, default=None, help='evaluate only the first N test articles (quick check)')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--bertscore', action='store_true', help='also compute BERTScore (pip install bert-score)')
    args = ap.parse_args()

    if args.baselines:
        run_baselines()
        return

    print_gpu()
    print_section(f"Evaluation on TEST: {args.model}")
    path = model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = MT5ForConditionalGeneration.from_pretrained(path)
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    tasks = [t.strip() for t in args.tasks.split(',')]
    results = {'model': args.model, 'num_beams': NUM_BEAMS}

    if 'summ' in tasks:
        df = load_split('test', 'summ')
        if args.max_summ:
            df = df.head(args.max_summ)
        if config.get('smoke_rows'):
            df = df.head(config['smoke_rows'])
        preds = generate(model, tokenizer, list(df['text']), config['data']['max_summary_length'],
                         NUM_BEAMS, args.batch_size, 'Summarization')
        r = rouge(preds, list(df['target']))
        r.update({'n': len(df),
                  'pred_words_mean': round(float(np.mean([len(p.split()) for p in preds])), 1),
                  'ref_words_mean': round(float(np.mean([len(t.split()) for t in df['target']])), 1)})
        if args.bertscore:
            from bert_score import score as bscore
            _, _, f1 = bscore(preds, list(df['target']), lang='fa', batch_size=32)
            r['bertscore_f1'] = round(100 * float(f1.mean()), 2)
        results['summarization'] = r
        pd.DataFrame({'article': [t[:400] for t in df['text']], 'reference': df['target'], 'prediction': preds}) \
            .to_csv(f"{RESULTS_DIR}{args.model}_summ_predictions.csv", index=False, encoding='utf-8-sig')
        print("📊 ROUGE (F1 x 100):", r)

    if 'sent' in tasks:
        df = load_split('test', 'sent')
        if config.get('smoke_rows'):
            df = df.head(config['smoke_rows'])
        raw = generate(model, tokenizer, list(df['text']), 8, 1, args.batch_size * 2, 'Sentiment')
        preds = [parse_sentiment(x) for x in raw]
        m = sentiment_report(list(df['target']), preds)
        results['sentiment'] = m
        pd.DataFrame({'review': df['text'], 'label': df['target'], 'raw_output': raw, 'prediction': preds}) \
            .to_csv(f"{RESULTS_DIR}{args.model}_sent_predictions.csv", index=False, encoding='utf-8-sig')
        neg = m['per_class']['منفی']
        print(f"📊 Accuracy={m['accuracy']:.3f}  Macro-F1={m['macro_f1']:.3f}  Weighted-F1={m['weighted_f1']:.3f}")
        print(f"   Negative recall={neg['recall']:.3f} ({round(neg['recall'] * neg['support'])} of {neg['support']})"
              f"   Invalid outputs={m['invalid_outputs']}")

    if 'loss' in tasks:
        # Same loss, same mode (no dropout), on training and validation samples -> directly comparable
        summ_len = config['data']['max_summary_length']
        results['loss_check'] = {'note': 'mean cross-entropy in eval mode on up to 500 random samples per split'}
        for task in ('summ', 'sent'):
            results['loss_check'][task] = {
                'train': mean_loss(model, tokenizer, load_split('train', task), summ_len),
                'val': mean_loss(model, tokenizer, load_split('val', task), summ_len),
            }
        print("📊 Loss check:", results['loss_check'])

    save(args.model, results)


if __name__ == "__main__":
    main()
