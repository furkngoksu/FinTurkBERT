# Data and experiment notes

## Research data

The report describes financial news collected from approximately twenty Turkish sources, filtered and deduplicated before masked language model pretraining. The sentiment dataset combines translated Financial PhraseBank (4,846 examples), curated financial corpus sentences (2,197), and translated FiQA (771), totaling 7,814 examples.

The full news corpus and labeled research split are stored separately from this repository. The generators below provide small synthetic datasets for trying the evaluation tools.

## Synthetic diagnostics

`evaluation/generate_benchmark_dataset.py` and `evaluation/generate_phrasebank_synthetic.py` generate scenario-based synthetic examples when run locally. Generated datasets, evaluation reports, and downloaded model weights are not included in the repository. The synthetic examples are separate from the research test split.

## Research files

- `pretraining/DL_project_splitting.ipynb`: the original DAPT experiment notebook.
- `training/all_code.py`: the original LoRA sentiment fine-tuning script; it expects a local `config.yaml` and labeled JSONL data.
- `training/tr_finance_pipeline_demo.py`: financial sentence curation and sampling.
- `training/translator.py`: financial text localization using the OpenAI API.

Sentiment training records use `text` and integer `label` fields: negative = 0, neutral = 1, positive = 2. The full research datasets and experiment configurations are not bundled with the repository. The README reports the original experiment results.
