# Joint Persian Abstractive Summarization and Sentiment Analysis with a Single mT5 Model

Code for the paper *"Joint Persian Abstractive Summarization and Sentiment Analysis with a Single mT5 Model: A Text-to-Text Multi-Task Study"* (Pourhatami, Amini, Khalilian).

One **mT5-base** model is fine-tuned for two tasks, each selected by a Persian task prefix:

| Task | Input | Target |
|---|---|---|
| Abstractive summarization | `خلاصه کن: <news article>` | reference summary |
| Sentiment (3 classes) | `احساس: <product review>` | `مثبت` / `خنثی` / `منفی` |

Training follows three stages: **(1)** summarization from mT5-base → **(2)** sentiment from the Stage 1 model → **(3)** shuffled mixture of both tasks from the Stage 2 model.

> The two corpora are not aligned: news articles have reference summaries, reviews have sentiment labels. No document has both, so each training example belongs to exactly one task.

## Data

| Dataset | Task | Source | Used |
|---|---|---|---|
| PN-Summary | Summarization | [HooshvareLab/pn_summary](https://huggingface.co/datasets/HooshvareLab/pn_summary) · [GitHub](https://github.com/hooshvare/pn-summary) | Official split: train sample 15,007 · dev · **test 5,593** |
| Digikala comments | Sentiment | [Kaggle: soheiltehranipour/digikala-comments-persian-sentiment-analysis](https://www.kaggle.com/datasets/soheiltehranipour/digikala-comments-persian-sentiment-analysis) (`data.csv`) | 2,991 unique reviews · stratified 2,093 / 449 / **449** |

Sentiment labels come from the 0–100 rating: positive ≥ 80, neutral 50–79, negative < 50.

The processed files used in the paper are included as zip archives (`summ_clean.zip`, `sent_clean.zip`, `joint_*.zip`).
To rebuild them from the raw data, put `digikala_comments.csv` in `data/raw/`, `pn_summary.zip` in the repository root, and run steps 1–3.

## Files

| Step | File | What it does | GPU |
|---|---|---|---|
| 1 | `1_analyze_datasets.py` | Dataset statistics (class distribution, lengths in characters and words) | no |
| 2 | `2_preprocess.py` | Normalization, cleaning, deduplication; keeps the official PN-Summary train/dev/test split | no |
| 3 | `3_prepare_joint_data.py` | Adds task prefixes, splits Digikala 70/15/15 (stratified), writes `joint_train/val/test.csv`, checks train–test overlap | no |
| 4 | `4_train_stage1_summ.py` | Stage 1: summarization (mT5-base → `summ_final`) | yes |
| 5 | `5_train_stage2_sent.py` | Stage 2: sentiment (`summ_final` → `sent_final`) | yes |
| 6 | `6_train_stage3_joint.py` | Stage 3: both tasks (`sent_final` → `joint_final`) | yes |
| 7 | `7_evaluate.py` | Test-set evaluation: ROUGE-1/2/L, accuracy, P/R/F1, macro-F1, confusion matrix; baselines | yes* |
| 8 | `8_plot_results.py` | Loss curves, confusion matrices and the results table from real outputs | no |
| 9 | `9_ablation.py` | Ablations: sentiment only from mT5-base; joint training without the curriculum | yes |
| – | `train_common.py` | Shared training code used by steps 5, 6 and 9 | – |
| – | `run_colab.ipynb` | Runs everything in Google Colab | – |

\* `python 7_evaluate.py --baselines` needs no GPU.

## Run in Google Colab

Open `run_colab.ipynb` from the **GitHub** tab of Colab, choose **Runtime → Change runtime type → T4 GPU**, and run the cells in order. Or, from a terminal:

```bash
pip install rouge_score sentencepiece
python 4_train_stage1_summ.py
python 5_train_stage2_sent.py
python 6_train_stage3_joint.py
python 7_evaluate.py --baselines
python 7_evaluate.py --model summ_final  --tasks summ
python 7_evaluate.py --model sent_final  --tasks summ,sent
python 7_evaluate.py --model joint_final --tasks summ,sent,loss
python 9_ablation.py          # optional
python 8_plot_results.py
```

Outputs: models in `outputs/models/`, metrics and predictions in `outputs/results/`, figures in `outputs/figures/`.

## Settings

mT5-base · AdamW, learning rate 5e-5, weight decay 0.01, 500 warm-up steps · batch 2 × 4 gradient-accumulation steps (effective 8) · at most 3 epochs per stage with early stopping on validation loss · max. input 512 / target 128 tokens · fp32 · no label smoothing · seed 42 · evaluation with beam search (4 beams, no repeated trigrams) · ROUGE with whitespace tokenization and no stemming (the default `rouge_score` tokenizer removes Persian letters).

Tested on one NVIDIA Tesla T4 (16 GB) in Google Colab.

## Citation

```bibtex
@article{pourhatami_joint_persian_mt5,
  title   = {Joint Persian Abstractive Summarization and Sentiment Analysis with a Single mT5 Model: A Text-to-Text Multi-Task Study},
  author  = {Pourhatami, Zahra and Amini, Amineh and Khalilian, Majid},
  journal = {SN Computer Science},
  note    = {under review}
}
```

## License

Code: MIT. Datasets keep the licenses of their original sources.
