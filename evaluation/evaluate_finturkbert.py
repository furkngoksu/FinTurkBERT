#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

from finturkbert_utils import (
    LABELS,
    load_model,
    predict_texts,
    tr_category,
    tr_difficulty,
    tr_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FinTurkBERT against a labeled benchmark."
    )
    parser.add_argument(
        "--dataset",
        default="data/finturkbert_benchmark.jsonl",
        help="Path to the JSONL benchmark dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default="models",
        help="Directory containing the local Hugging Face cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Maximum token length passed to the model.",
    )
    parser.add_argument(
        "--predictions-out",
        default="reports/predictions.csv",
        help="CSV file with row-level predictions.",
    )
    parser.add_argument(
        "--summary-out",
        default="reports/evaluation_summary.json",
        help="JSON file with aggregate metrics.",
    )
    parser.add_argument(
        "--mismatches-out",
        default="reports/mismatches.csv",
        help="CSV file containing only incorrect predictions.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def round_metric(value: float) -> float:
    return round(value, 4)


def compute_metrics(rows: list[dict]) -> dict:
    confusion = {
        true_label: {pred_label: 0 for pred_label in LABELS}
        for true_label in LABELS
    }
    for row in rows:
        confusion[row["ground_truth"]][row["predicted_label"]] += 1

    total = len(rows)
    correct = sum(
        confusion[label][label]
        for label in LABELS
    )
    per_class = {}
    f1_values = []

    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        support = sum(confusion[label].values())
        per_class[label] = {
            "precision": round_metric(precision),
            "recall": round_metric(recall),
            "f1": round_metric(f1),
            "support": support,
        }
        if support > 0:
            f1_values.append(f1)

    return {
        "count": total,
        "accuracy": round_metric(safe_divide(correct, total)),
        "macro_f1": round_metric(sum(f1_values) / len(f1_values)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def compute_group_metrics(rows: list[dict], group_key: str) -> dict[str, dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    return {
        key: compute_metrics(group_rows)
        for key, group_rows in sorted(groups.items())
    }


def attach_predictions(dataset_rows: list[dict], prediction_rows: list[dict]) -> list[dict]:
    enriched_rows = []
    for data_row, pred_row in zip(dataset_rows, prediction_rows):
        enriched_rows.append(
            {
                **data_row,
                "ground_truth_tr": data_row.get(
                    "ground_truth_tr",
                    tr_label(data_row["ground_truth"]),
                ),
                "difficulty_tr": data_row.get(
                    "difficulty_tr",
                    tr_difficulty(data_row["difficulty"]),
                ),
                "category_tr": data_row.get(
                    "category_tr",
                    tr_category(data_row["category"]),
                ),
                "predicted_label": pred_row["label"],
                "predicted_label_tr": pred_row.get(
                    "label_tr",
                    tr_label(pred_row["label"]),
                ),
                "score_negative": pred_row["scores"]["negative"],
                "score_neutral": pred_row["scores"]["neutral"],
                "score_positive": pred_row["scores"]["positive"],
                "correct": data_row["ground_truth"] == pred_row["label"],
            }
        )
    return enriched_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def print_summary(summary: dict, mismatches: list[dict]) -> None:
    print(
        f"Accuracy: {summary['overall']['accuracy']:.4f} | "
        f"Macro-F1: {summary['overall']['macro_f1']:.4f} | "
        f"Count: {summary['overall']['count']}"
    )
    print("Per-class metrics:")
    for label in LABELS:
        metrics = summary["overall"]["per_class"][label]
        print(
            f"  {tr_label(label):<8} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"support={metrics['support']}"
        )

    print("Confusion matrix:")
    for true_label in LABELS:
        print(
            f"  {tr_label(true_label):<8} "
            f"{summary['overall']['confusion_matrix'][true_label]}"
        )

    print("By difficulty:")
    for difficulty, metrics in summary["by_difficulty"].items():
        print(
            f"  {tr_difficulty(difficulty):<8} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} count={metrics['count']}"
        )

    print(f"Mismatches: {len(mismatches)}")
    for row in mismatches[:8]:
        print(
            f"  {row['id']} gt={row['ground_truth_tr']} pred={row['predicted_label_tr']} "
            f"| {row['text']}"
        )


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset_rows = load_dataset(dataset_path)
    texts = [row["text"] for row in dataset_rows]

    tokenizer, model = load_model(args.cache_dir)
    prediction_rows = predict_texts(
        texts,
        tokenizer,
        model,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    enriched_rows = attach_predictions(dataset_rows, prediction_rows)
    mismatches = [row for row in enriched_rows if not row["correct"]]

    summary = {
        "dataset": str(dataset_path),
        "cache_dir": args.cache_dir,
        "overall": compute_metrics(enriched_rows),
        "by_difficulty": compute_group_metrics(enriched_rows, "difficulty"),
        "by_category": compute_group_metrics(enriched_rows, "category"),
        "mismatch_count": len(mismatches),
    }

    write_csv(Path(args.predictions_out), enriched_rows)
    write_csv(Path(args.mismatches_out), mismatches)
    write_json(Path(args.summary_out), summary)
    print_summary(summary, mismatches)


if __name__ == "__main__":
    main()
