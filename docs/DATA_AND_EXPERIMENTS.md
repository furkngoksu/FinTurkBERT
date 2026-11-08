# Data and experiment notes

## Research data

The report describes financial news collected from approximately twenty Turkish sources, filtered and deduplicated before masked language model pretraining. The sentiment dataset combines translated Financial PhraseBank (4,846 examples), curated financial corpus sentences (2,197), and translated FiQA (771), totaling 7,814 examples.

The full news corpus and labeled research split are stored separately from this repository. The generators below provide small synthetic datasets for trying the evaluation tools.

## Synthetic diagnostics

`evaluation/generate_benchmark_dataset.py` and `evaluation/generate_phrasebank_synthetic.py` generate scenario-based synthetic examples locally. CSV and JSONL outputs are ignored by Git. These are separate from the research test split; their metrics are not the paper's test results.

## Working directories

- `evaluation/data/`: generated diagnostics and local inputs.
- `evaluation/reports/`: predictions, mismatches, metrics, HTML reports, and figures.
- `evaluation/models/`: downloaded checkpoints.

Evaluation commands write their outputs to these directories. Keep source datasets separate from generated predictions and reports.

## Training and evaluation

Fine-tuning is configured in `configs/finetune.yaml` and run with `training/train.py`. Supply a JSONL sentiment dataset with `text` and integer `label` fields: negative = 0, neutral = 1, positive = 2. The script creates stratified training, validation, and test splits with a fixed seed. Optional task-adaptive pretraining uses only the training split. The training script reports weighted F1; the diagnostic evaluator reports macro F1 across the three sentiment classes.

`pretraining/domain_pretraining.ipynb` expects a JSON array of news documents with `content` and `url` fields. Run it from `pretraining/`; the default input is `../data/financial_news.json`. It splits documents into 90% training, 5% validation, and 5% test before sentence segmentation and packing. Validation is used for model selection; test data is reserved for final evaluation. Paired MLM evaluations reset the random seed to use matching masks.

The README results describe the original experiments, using three epochs, LoRA rank 16, and dropout 0.20. Current scripts include subsequent workflow corrections. Reproducing the reported scores requires the original data, splits, and checkpoints; new runs should be reported separately.
