#!/usr/bin/env python3

import argparse

from finturkbert_utils import load_model, predict_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and run the FinTurkBERT sentiment model locally."
    )
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Input sentence to score. Repeat --text for multiple inputs.",
    )
    parser.add_argument(
        "--cache-dir",
        default="models",
        help="Directory used to store the downloaded Hugging Face model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="Maximum token length passed to the model.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.cache_dir)
    for idx, row in enumerate(
        predict_texts(args.text, tokenizer, model, max_length=args.max_length),
        start=1,
    ):
        print(f"[{idx}] {row['text']}")
        print(f"label: {row['label']}")
        print(f"scores: {row['scores']}")
        print()


if __name__ == "__main__":
    main()
