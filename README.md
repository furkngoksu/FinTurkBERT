# FinTurkBERT

Domain adaptation and sentiment analysis for Turkish financial text. This collaborative project adapts BERTurk to financial language through continued pretraining, then fine-tunes it with LoRA to classify negative, neutral, and positive sentiment.

**Authors:** Deniz Topal, Furkan Yasir Göksu, Faruk Akyol — Istanbul Technical University.

## Research pipeline

1. **Corpus preparation:** collect Turkish financial news, filter financial content, remove exact and near duplicates, and pack complete sentences into training sequences.
2. **Domain adaptation:** continue masked language model pretraining on the financial corpus and evaluate on Turkish Wikipedia to examine general-language retention.
3. **Sentiment data:** translate and financially localize Financial PhraseBank and FiQA examples, and curate additional annotated Turkish financial sentences.
4. **Fine-tuning:** use LoRA and compare domain-adapted checkpoints with general Turkish BERT models.

## Reported results

The report uses a sentiment dataset of 7,814 labeled sentences. Under the same regularized configuration (LoRA rank 16, dropout 0.20, three epochs), it reports:

| Model | Test accuracy | Test F1 |
| --- | ---: | ---: |
| BERTurk, uncased | 80.59% | 79.17% |
| BERTurk, cased | 81.82% | 81.83% |
| FinTurkBERT DAPT v2 | 84.40% | 83.52% |
| FinTurkBERT DAPT v3 | 82.71% | 82.73% |

DAPT v2 improves accuracy by 2.58 percentage points over cased BERTurk in this configuration. These are the report's held-out test results; the synthetic diagnostics generated here are separate evaluations.

## Repository layout

```text
dl-dapt/         Corpus preparation and DAPT experiment notebook
dl-fine-tune/    Training, translation, and financial data preparation scripts
evaluation/     Inference, synthetic diagnostics, metrics, and HTML report tools
docs/           Presentation and research notes
```

Datasets, checkpoints, logs, and generated reports are kept locally and excluded from version control. Existing script directories are retained so relative paths remain compatible.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA environment. Translation scripts require your own `OPENAI_API_KEY` environment variable. Training scripts retain experiment-specific input paths; configure these for your local datasets before running them.

## Run sentiment inference

The utility downloads `ff112/FinTurkBERT` from Hugging Face:

```bash
python evaluation/run_finturkbert.py \
  --cache-dir evaluation/models \
  --max-length 96 \
  --text "Şirketin net kârı geçen yılın aynı dönemine göre arttı."
```

## Generate and evaluate synthetic diagnostics

Run from `evaluation/` so default relative paths resolve correctly:

```bash
cd evaluation
python generate_benchmark_dataset.py
python evaluate_finturkbert.py
python build_html_report.py
```

This creates local `data/` and `reports/` directories. A second generator is `generate_phrasebank_synthetic.py`. Both produce synthetic examples, not the original training corpus or the paper's held-out test set.

See [data and experiment notes](docs/DATA_AND_EXPERIMENTS.md) for provenance and reproduction limits. The presentation is available in [docs](docs/).
