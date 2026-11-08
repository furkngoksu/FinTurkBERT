# Data and experiment notes

## Research data

The report describes financial news collected from approximately twenty Turkish sources, filtered and deduplicated before masked language model pretraining. The sentiment dataset combines translated Financial PhraseBank (4,846 examples), curated financial corpus sentences (2,197), and translated FiQA (771), totaling 7,814 examples.

The full research datasets are not distributed here. Obtain source datasets from their providers and review their terms before reuse or redistribution. This repository does not grant rights to third-party news content or source datasets.

## Synthetic diagnostics

`evaluation/generate_benchmark_dataset.py` and `evaluation/generate_phrasebank_synthetic.py` generate scenario-based synthetic examples locally. CSV and JSONL outputs are ignored by Git. These are separate from the research test split; their metrics are not the paper's test results.

## Local artifacts

- `evaluation/data/`: generated diagnostics and local inputs.
- `evaluation/reports/`: predictions, mismatches, metrics, HTML reports, and figures.
- `evaluation/models/`: downloaded checkpoints.

Logs and archived outputs are also excluded. Cleanup removes previously tracked generated files from the current tree; earlier versions remain recoverable from Git history. Historical commits are unchanged.

## Reproduction limits

Training code retains experiment-specific paths and configuration. Reproducing the report requires the original preprocessing, split, checkpoints, and settings; generating synthetic diagnostics alone does not reproduce those results.

The README table transcribes the report's three-epoch, rank-16, dropout-0.20 comparison. Higher-capacity runs use different settings and should not be mixed into that comparison. DAPT version labels should not be inferred from corpus sizes alone.
