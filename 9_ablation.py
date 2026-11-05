"""
Ablation study (real runs, replaces the hard-coded numbers of the old version)

  python 9_ablation.py                 # runs both ablations, then evaluates them on the test split
  python 9_ablation.py --only sent     # only the sentiment-only model   (about 20-30 min on a T4)
  python 9_ablation.py --only joint    # only the joint-without-curriculum model (several hours)

What is compared (all evaluated on the same held-out test sets with 7_evaluate.py):

  Model                                   How it is trained                          Answers
  summ_final   (Stage 1)                  mT5-base -> summarization                  summarization alone
  sent_final   (Stages 1->2)              Stage 1  -> sentiment                      does Stage 1 help sentiment?
  joint_final  (Stages 1->2->3)           Stage 2  -> mixed tasks       = full model
  sent_only    (ablation A)               mT5-base -> sentiment                      sentiment without any summarization training
  joint_only   (ablation B)               mT5-base -> mixed tasks                    joint training without the curriculum

Compare  sent_only  vs  sent_final / joint_final  -> effect of the curriculum on sentiment
Compare  joint_only vs  joint_final               -> effect of Stages 1 and 2 on both tasks
Compare  summ_final vs  joint_final               -> does later training hurt summarization?
"""
import argparse
import subprocess
import sys


def run(cmd):
    print("\n" + "=" * 70 + f"\n▶ {' '.join(cmd)}\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['sent', 'joint'], default=None)
    args = ap.parse_args()

    if args.only in (None, 'sent'):
        run(['5_train_stage2_sent.py', '--init', 'base', '--name', 'sent_only'])
        run(['7_evaluate.py', '--model', 'sent_only', '--tasks', 'sent'])

    if args.only in (None, 'joint'):
        run(['6_train_stage3_joint.py', '--init', 'base', '--name', 'joint_only'])
        run(['7_evaluate.py', '--model', 'joint_only', '--tasks', 'summ,sent'])

    run(['8_plot_results.py'])


if __name__ == "__main__":
    main()
