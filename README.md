# FinTurkBERT: Turkish Financial Sentiment Analysis

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <a href="https://huggingface.co/ff112/FinTurkBERT"><img alt="Hugging Face Model" src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" /></a>
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-FF5722?style=for-the-badge&logo=huggingface&logoColor=white" />
</p>

[Model & model card](https://huggingface.co/ff112/FinTurkBERT) · [Pretraining notebook](pretraining/DL_project_splitting.ipynb) · [Presentation](docs/FinTurkBERT-Presentation.pptx)

FinTurkBERT is a project for analyzing sentiment in Turkish financial text. It classifies statements from financial news, company announcements, and market commentary as positive, negative, or neutral from an investor's perspective.

The focus is on the financial meaning of a statement, including changes in business performance, earnings, and risk. FinTurkBERT can support financial news monitoring, company disclosure analysis, and research on Turkish financial language.

The model adapts BERTurk to financial text through domain-adaptive pretraining and LoRA fine-tuning.

This repository contains the pretraining notebook, data preparation and fine-tuning scripts, and tools for running sentiment predictions and evaluation.

## Installation

Use Python 3.10 or newer. From a terminal:

```bash
git clone https://github.com/furkngoksu/FinTurkBERT.git
cd FinTurkBERT
python -m venv .venv
source .venv/bin/activate
python -m pip install torch transformers
```

These dependencies cover the prediction and evaluation tools. See [research files and data](docs/DATA_AND_EXPERIMENTS.md) for the training scripts and notebook setup.

## Sentiment prediction

From the repository root:

```bash
python evaluation/run_finturkbert.py \
  --cache-dir evaluation/models \
  --max-length 96 \
  --text "Şirketin net kârı geçen yılın aynı dönemine göre arttı."
```

The script downloads the [ff112/FinTurkBERT](https://huggingface.co/ff112/FinTurkBERT) checkpoint and prints a sentiment label and class probabilities. Repeat `--text` to process multiple statements.

## Evaluation

From the repository root, generate a synthetic dataset and evaluate the model:

```bash
cd evaluation
python generate_benchmark_dataset.py
python evaluate_finturkbert.py
```

The evaluator writes predictions, incorrect predictions, and aggregate metrics locally. To create an HTML report from a run containing incorrect predictions:

```bash
python build_html_report.py
```

The report is written to `reports/report.html`, relative to `evaluation/`. These synthetic evaluations are separate from the research test set used for the results below.

## Results

The project experiments reported the following held-out sentiment classification results. Each row uses three epochs, LoRA rank 16, and dropout 0.20.

| Model | Accuracy (%) | F1 (%) |
| --- | ---: | ---: |
| BERTurk, uncased | 80.59 | 79.17 |
| BERTurk, cased | 81.82 | 81.83 |
| FinTurkBERT DAPT v2 | 84.40 | 83.52 |
| FinTurkBERT DAPT v3 | 82.71 | 82.73 |

This table compares research checkpoints. The downloadable checkpoint's [model card](https://huggingface.co/ff112/FinTurkBERT#evaluation-summary) reports a separate validation result.

## Repository

| Directory | Contents |
| --- | --- |
| [pretraining](pretraining/) | DAPT experiment notebook |
| [training](training/) | Financial text curation, localization, and sentiment fine-tuning scripts |
| [evaluation](evaluation/) | Model inference, synthetic data generators, metrics, and report tools |
| [docs](docs/) | Project presentation and data notes |

The research datasets and training configurations are not included. Data sources and file requirements are described in [the data notes](docs/DATA_AND_EXPERIMENTS.md). The [project presentation](docs/FinTurkBERT-Presentation.pptx) is also available.

## Authors

Deniz Topal, Furkan Yasir Göksu, and Faruk Akyol — Istanbul Technical University.
