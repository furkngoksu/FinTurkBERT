# FinTurkBERT

FinTurkBERT is a Turkish financial language model developed for sentiment analysis. It adapts BERTurk to financial news through domain-adaptive pretraining, followed by task-specific fine-tuning with LoRA. The model classifies financial statements as positive, negative, or neutral from an investor's perspective.

**Authors:** Deniz Topal, Furkan Yasir Göksu, Faruk Akyol — Istanbul Technical University.

## Overview

The project explores how domain-specific data and continued pretraining improve financial language understanding in Turkish. It covers the complete workflow from corpus preparation to sentiment classification and evaluation.

- **Financial corpus:** news collected from Turkish financial sources, cleaned through domain filtering, deduplication, and sentence-level preprocessing.
- **Domain adaptation:** masked language model pretraining on financial text, with additional evaluation on Turkish Wikipedia to assess general-language retention.
- **Sentiment dataset:** 7,814 labeled sentences combining localized Financial PhraseBank and FiQA examples with curated Turkish financial text.
- **Model training:** LoRA fine-tuning and experiments on model capacity and regularization.

## Results

FinTurkBERT DAPT v2 achieved **84.40% test accuracy** and **83.52 F1**, compared with **81.82% accuracy** for cased BERTurk under the same fine-tuning configuration.

| Model | Test accuracy | Test F1 |
| --- | ---: | ---: |
| BERTurk, uncased | 80.59% | 79.17% |
| BERTurk, cased | 81.82% | 81.83% |
| FinTurkBERT DAPT v2 | 84.40% | 83.52% |
| FinTurkBERT DAPT v3 | 82.71% | 82.73% |

All rows use LoRA rank 16, dropout 0.20, and three training epochs. These results come from the project report's held-out sentiment test set. The repository also provides separate synthetic evaluation tools for examining model behavior.

## Repository layout

```text
dl-dapt/         Corpus preparation and DAPT experiment notebook
dl-fine-tune/    Training, translation, and financial data preparation scripts
evaluation/     Inference, synthetic diagnostics, metrics, and HTML report tools
docs/           Presentation and research notes
```

Datasets, model checkpoints, and generated outputs are excluded from version control. See the [data and experiment notes](docs/DATA_AND_EXPERIMENTS.md) for data sources and reproduction details.

## Getting started

Use Python 3.10 or newer:

```bash
git clone https://github.com/furkngoksu/FinTurkBERT.git
cd FinTurkBERT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your CUDA environment. Translation scripts use an `OPENAI_API_KEY` environment variable. Training scripts require local datasets and experiment-specific paths to be configured before use.

## Run sentiment inference

Run the published [FinTurkBERT checkpoint](https://huggingface.co/ff112/FinTurkBERT) on a Turkish financial statement:

```bash
python evaluation/run_finturkbert.py \
  --cache-dir evaluation/models \
  --max-length 96 \
  --text "Şirketin net kârı geçen yılın aynı dönemine göre arttı."
```

## Evaluation

Generate a synthetic benchmark, run model predictions, and produce an HTML report:

```bash
cd evaluation
python generate_benchmark_dataset.py
python evaluate_finturkbert.py
python build_html_report.py
```

The commands run from `evaluation/` and write to local `data/` and `reports/` directories. `generate_phrasebank_synthetic.py` provides an additional set of financial sentiment scenarios. These synthetic datasets are intended for diagnostic evaluation and are separate from the research test set.

## Documentation

- [Project presentation](docs/FinTurkBERT-Presentation.pptx)
- [Data sources and experiment notes](docs/DATA_AND_EXPERIMENTS.md)
