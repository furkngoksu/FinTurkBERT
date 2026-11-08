# Training and data preparation

Run commands from the repository root unless noted otherwise. Keep inputs under `data/` and generated training files under `outputs/`.

## Financial sentence curation

The curation tool reads JSONL or CSV news records, selects sentences using financial lexicons, and optionally downsamples and filters the candidates:

```bash
python training/prepare_corpus.py run-all \
  --input data/news.jsonl --format jsonl --text-field content \
  --workdir outputs/curation --lex-dir outputs/lexicons \
  --total-target 20000 --max-per-template 3 --min-true 4
```

The output contains candidate sentences, feature flags, and source information. This stage does not assign sentiment labels. It uses one worker by default for reproducible sampling. Optional `--use-stanza` and `--use-morph` require Stanza and Zeyrek respectively; install those packages separately if using these options.

## Financial localization

The translation tool localizes English financial text into Turkish and adds `translated_text` to each JSONL record. It uses the OpenAI API, so running it sends the selected text to the service and incurs API usage charges. Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` in your environment before running:

```bash
python training/translator.py data/english_finance.jsonl \
  --output outputs/translated_finance.jsonl --number 10
```

The input is preserved. Output records retain the original fields and labels. Currency localization uses the script's fixed annual conversion table; entity substitutions are generated adaptations rather than factual news reports.

## Sentiment fine-tuning

Prepare a JSONL file with `text` and `label` fields and configure its path in `configs/finetune.yaml`. The example configuration starts from BERTurk; replace `model.name` with a DAPT checkpoint for domain-adapted training.

```bash
python training/train.py --config configs/finetune.yaml
```

The script creates stratified 80/10/10 splits, uses validation weighted F1 for checkpoint selection, and writes test metrics separately. Optional TAPT continues masked language modeling on training texts only. The saved model contains LoRA adapters and a classification head; load it using PEFT's sequence-classification model loader.

The published checkpoint used by `evaluation/run_finturkbert.py` is separate from newly trained local adapters. Original report scores describe the historical experiments; new runs with the corrected workflow should be evaluated and reported independently.
