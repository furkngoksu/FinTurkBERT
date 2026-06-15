# 📈 FinTurkBERT: Domain-Adaptive Pretraining and Sentiment Analysis for Turkish Financial Texts

FinTurkBERT is a domain-specific Natural Language Processing (NLP) project that implements **Domain-Adaptive Pretraining (DAPT)** and **Task-Adaptive Pretraining (TAPT)** on Turkish financial news, followed by sentiment analysis fine-tuning using Low-Rank Adaptation (LoRA).

---

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Transformers-FF5722?style=for-the-badge&logo=huggingface&logoColor=white" />
</p>

---

## 📂 Project Structure

```
├── docs/
│   ├── FinTurkBERT-Presentation.pptx          # Project presentation slides
│   └── FinTurkBERT-Report.pdf                # Comprehensive academic report
├── dl-dapt/
│   └── DL_project_splitting.ipynb            # Dataset splitting & DAPT pre-processing (clean outputs)
├── dl-fine-tune/
│   ├── all_code.py                           # Full pipeline code for fine-tuning
│   ├── tr_finance_pipeline_demo.py           # Demo pipeline execution script
│   └── translator.py                         # LLM translation utilities for financial localization
├── evaluation/
│   ├── build_html_report.py                  # HTML report generator script
│   ├── evaluate_finturkbert.py               # Evaluation runner for FinTurkBERT
│   ├── finturkbert_utils.py                  # Utility functions for evaluation
│   ├── generate_benchmark_dataset.py         # Benchmark dataset creation utility
│   ├── generate_phrasebank_synthetic.py      # Synthetic data generator using GPT-4o
│   ├── run_finturkbert.py                    # Sentiment classifier demo runner
│   ├── RESULTS-BERTS.txt                     # Performance logs for models (redacted tokens)
│   ├── last_log.txt                          # Evaluation execution logs
│   ├── llm_results.zip                       # Zero-shot evaluation outputs from LLMs
│   ├── data/                                 # Evaluation benchmark and synthetic datasets
│   └── reports/                              # Generated HTML evaluation reports and SVG graphs
├── requirements.txt                          # List of python package dependencies
└── README.md                                 # Project documentation
```

---

## 🚀 Key Project Stages

### 1. Domain-Adaptive Pretraining (DAPT) (`dl-dapt/`)
- A base Turkish BERT model (**BERTurk**) is adapted to the financial domain using an unlabeled corpus built from Turkish financial news websites.
- **Pretraining Objective**: Masked Language Modeling (MLM).
- **Catastrophic Forgetting Mitigation**: Evaluated on Turkish Wikipedia splits to ensure general language understanding is preserved while specializing in financial terminology.

### 2. Fine-Tuning & Financial Localization (`dl-fine-tune/`)
- **Financial Localization Pipeline**: English sentiment analysis datasets (FIQA and Financial PhraseBank) are translated and localized into Turkish using a two-stage LLM pipeline (GPT-4o) ensuring currency conversions (USD ➔ TRY) and natural financial phrasing.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Built with **LoRA (Low-Rank Adaptation)** on attention layers to optimize training parameters (~0.3% trainable parameters) and avoid overfitting on a RTX 4060 GPU.

### 3. Evaluation & Results (`evaluation/`)
- Models are benchmarked across multiple configurations (Baseline BERTurk vs. FinTurkBERT variants, LoRA ranks, and Zero-shot LLMs).
- FinTurkBERT DAPT v2 achieves **84.40% test accuracy** and **83.52% F1 score** on Turkish financial sentiment tasks, significantly outperforming baseline LLMs (Qwen, Llama, Gemma, Mistral) and general-purpose BERTurk models.
- Interactive HTML reports and performance visualization graphs are available under [evaluation/reports/](evaluation/reports/).

---

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/furkngoksu/FinTurkBERT.git
   cd FinTurkBERT
   ```

2. **Install Dependencies**
   Ensure Python 3.8+ and PyTorch (with GPU support if training) are set up. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Explore Documentation**
   Academic report details, dataset token distributions, and methodology can be found under the [docs/](docs/) folder.
