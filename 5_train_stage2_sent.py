"""
Stage 2 - Sentiment training (fixed version)

  python 5_train_stage2_sent.py                                  # normal: starts from the Stage 1 model
  python 5_train_stage2_sent.py --init base --name sent_only     # ablation: starts from plain mT5-base

Changes vs. previous version:
  1. Uses the Digikala rows of joint_train.csv / joint_val.csv (2,093 / 449).
     Previously sent_clean.csv was re-split, so test reviews could be used for training.
  2. Same prefix as the rest of the pipeline: "احساس: " (was "تحلیل احساس: ").
  3. Outputs that are not exactly one label are counted as errors, not silently set to "neutral".
  4. label_smoothing = 0; dynamic padding with -100 in the labels.
"""
import argparse
from utils import print_section
from train_common import config, load_split, train_seq2seq, print_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='summ_final', help="'base' or a saved model name (default: Stage 1 model)")
    ap.add_argument('--name', default='sent_final', help='output model name')
    args = ap.parse_args()

    print_gpu()
    print_section(f"Stage 2: Sentiment training  ->  {args.name}  (init: {args.init})")
    train_df = load_split('train', task='sent')
    val_df = load_split('val', task='sent')
    print("📊 Train class distribution:", train_df['target'].value_counts().to_dict())

    train_seq2seq(train_df, val_df, init=args.init, name=args.name,
                  epochs=config['model']['epochs_stage2'], eval_steps=50,
                  max_target_len=8, sentiment_metrics=True)


if __name__ == "__main__":
    main()
