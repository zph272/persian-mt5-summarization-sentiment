"""
Stage 3 - Joint (multi-task) training (fixed version)

  python 6_train_stage3_joint.py                                   # normal: starts from the Stage 2 model
  python 6_train_stage3_joint.py --init base --name joint_only     # ablation: no curriculum, from mT5-base

Changes vs. previous version:
  1. Starts from the Stage 2 model (previously it loaded the Stage 1 model, so Stage 2 was skipped).
  2. Trains on joint_train.csv: real news->summary and review->label pairs, each with its task prefix,
     shuffled together (7,000 + 2,093). No random pairing of reviews with unrelated news summaries.
  3. Validation on joint_val.csv (1,500 + 449). The test split is never used here.
     (Previously joint_clean.csv was re-split, so test samples could be trained on.)
  4. Uses all training rows (previously df.head(5000)).
  5. Task balance: the 2,093 sentiment reviews are repeated 3 times (6,279 vs 7,000 summaries), and the
     loss gives every example the same weight (mean over its own target tokens, then mean over the batch).
     Otherwise long summaries dominate: with a token-level mean, sentiment labels (2-3 tokens) would make up
     only ~1% of the loss and the model could forget what it learned in Stage 2.
  6. Padding masked; no label smoothing.
"""
import pandas as pd
import argparse
from utils import print_section
from train_common import config, load_split, train_seq2seq, print_gpu

SENT_REPEAT = 3   # each sentiment review appears 3 times in Stage 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='sent_final', help="'base' or a saved model name (default: Stage 2 model)")
    ap.add_argument('--name', default='joint_final', help='output model name')
    args = ap.parse_args()

    print_gpu()
    print_section(f"Stage 3: Joint training  ->  {args.name}  (init: {args.init})")
    train_df = load_split('train')                      # both tasks
    sent = train_df[train_df['task'] == 'sent']
    train_df = pd.concat([train_df] + [sent] * (SENT_REPEAT - 1), ignore_index=True) \
                 .sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = load_split('val')

    train_seq2seq(train_df, val_df, init=args.init, name=args.name,
                  epochs=config['model']['epochs_joint'], eval_steps=300,
                  max_target_len=config['data']['max_summary_length'], sentiment_metrics=False,
                  per_example_loss=True)


if __name__ == "__main__":
    main()
