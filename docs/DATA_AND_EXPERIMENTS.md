# Research files and data

## Data

The pretraining corpus consists of Turkish financial news collected from approximately twenty sources and processed through filtering and deduplication. The sentiment dataset contains 7,814 labeled sentences:

| Source | Sentences |
| --- | ---: |
| Financial PhraseBank, translated and localized | 4,846 |
| Curated Turkish financial news | 2,197 |
| FiQA, translated and localized | 771 |

The full corpus, labeled research split, and training configuration are not included in this repository. Generated evaluation files and downloaded checkpoints are excluded from version control.

## Pretraining

[DL_project_splitting.ipynb](../pretraining/DL_project_splitting.ipynb) contains data preparation, token packing, domain-adaptive pretraining, and financial/Wikipedia MLM evaluation.

The notebook uses Google Colab and Google Drive. Set `CFG.data_path` to your corpus filename and adjust the Drive path in the data-loading cells. The expected input is a JSON array of documents with `content` and `url` fields.

For the notebook dependencies, run the following from the repository root in your Python environment:

```bash
python -m pip install -r requirements.txt "transformers>=4.41,<5"
python -m pip install nltk seaborn tqdm
```

## Sentiment fine-tuning

[all_code.py](../training/all_code.py) contains LoRA fine-tuning and an optional task-adaptive pretraining stage. It reads `config.yaml` from the working directory. The YAML must specify the model, data paths, LoRA settings, and training parameters used by the script.

Training data is JSONL with `text` and integer `label` fields: `0` = negative, `1` = neutral, `2` = positive. For example:

```json
{"text": "Şirketin net kârı arttı.", "label": 2}
```

With your configuration and dataset paths prepared, run from the repository root:

```bash
python -m pip install -r requirements.txt "transformers>=4.41,<5"
python training/all_code.py
```

## Data preparation

[tr_finance_pipeline_demo.py](../training/tr_finance_pipeline_demo.py) selects financial sentences, samples candidates, and filters them using lexicon features. Install `tqdm`, then inspect the available commands:

```bash
python -m pip install tqdm
python training/tr_finance_pipeline_demo.py --help
```

The `--use-stanza` and `--use-morph` options use optional Stanza and Zeyrek packages. Stanza also requires its Turkish language resources.

[translator.py](../training/translator.py) localizes English financial text into Turkish using the OpenAI API. It requires `OPENAI_API_KEY` and updates the supplied JSONL file in place. Use a working copy of the input when running it. API calls send the selected text to OpenAI and incur usage charges.

## Synthetic evaluation data

The two generators create examples from predefined financial scenarios:

- [generate_benchmark_dataset.py](../evaluation/generate_benchmark_dataset.py): financial sentiment scenarios with difficulty and category labels.
- [generate_phrasebank_synthetic.py](../evaluation/generate_phrasebank_synthetic.py): additional scenarios written from an investor's perspective.

Both accept `--samples-per-family`, `--jsonl-out`, and `--csv-out`. To evaluate the second generator's output, run from the repository root:

```bash
cd evaluation
python generate_phrasebank_synthetic.py
python evaluate_finturkbert.py --dataset data/phrasebank_viewpoint_synthetic.jsonl
```

These commands use the evaluator's default output paths and replace any previous prediction files at those paths. The generated examples are separate from the research test set; their evaluation scores should be interpreted separately from the README results.
