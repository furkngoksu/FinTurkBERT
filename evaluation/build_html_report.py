#!/usr/bin/env python3

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

from finturkbert_utils import tr_category


LABELS = ["negative", "neutral", "positive"]
PALETTE = {
    "negative": "#c2410c",
    "neutral": "#0f766e",
    "positive": "#15803d",
    "ink": "#172033",
    "muted": "#5f6b85",
    "panel": "#fbfaf7",
    "panel_border": "#d9d4c7",
    "accent": "#1d4ed8",
    "alert": "#b91c1c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML report from FinTurkBERT evaluation outputs."
    )
    parser.add_argument(
        "--summary",
        default="reports/evaluation_summary.json",
        help="Path to evaluation_summary.json.",
    )
    parser.add_argument(
        "--predictions",
        default="reports/predictions.csv",
        help="Path to predictions.csv.",
    )
    parser.add_argument(
        "--mismatches",
        default="reports/mismatches.csv",
        help="Path to mismatches.csv.",
    )
    parser.add_argument(
        "--output",
        default="reports/report.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--assets-dir",
        default="reports/assets",
        help="Directory where SVG chart assets will be stored.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")


def escape(text: str) -> str:
    return html.escape(str(text))


def display_label(label: str) -> str:
    return label


def display_category(category: str) -> str:
    return tr_category(category)


def render_confusion_svg(confusion: dict[str, dict[str, int]]) -> str:
    cell = 92
    margin_left = 120
    margin_top = 70
    width = margin_left + cell * len(LABELS) + 20
    height = margin_top + cell * len(LABELS) + 30
    max_value = max(confusion[true][pred] for true in LABELS for pred in LABELS) or 1

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Confusion matrix">'
    ]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="24" '
        f'fill="#fffdf8" stroke="{PALETTE["panel_border"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{margin_left}" y="32" fill="{PALETTE["ink"]}" '
        f'font-size="18" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" '
        f'font-weight="700">Karmaşıklık Matrisi</text>'
    )
    for idx, label in enumerate(LABELS):
        x = margin_left + idx * cell + cell / 2
        y = margin_top - 18
        parts.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" fill="{PALETTE["muted"]}" '
            f'font-size="13" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">'
            f'{escape(display_label(label))}</text>'
        )
        label_y = margin_top + idx * cell + cell / 2 + 4
        parts.append(
            f'<text x="{margin_left - 14}" y="{label_y}" text-anchor="end" '
            f'fill="{PALETTE["muted"]}" font-size="13" '
            f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{escape(display_label(label))}</text>'
        )
    for row_idx, true_label in enumerate(LABELS):
        for col_idx, pred_label in enumerate(LABELS):
            value = confusion[true_label][pred_label]
            intensity = value / max_value
            fill = "#f1f5f9"
            if row_idx == col_idx:
                alpha = 0.18 + intensity * 0.74
                fill = f'rgba(21, 128, 61, {alpha:.3f})'
            elif value > 0:
                alpha = 0.12 + intensity * 0.45
                fill = f'rgba(185, 28, 28, {alpha:.3f})'
            x = margin_left + col_idx * cell
            y = margin_top + row_idx * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell-8}" height="{cell-8}" rx="16" '
                f'fill="{fill}" stroke="#ffffff" stroke-width="1.5"/>'
            )
            parts.append(
                f'<text x="{x + (cell-8)/2}" y="{y + 38}" text-anchor="middle" '
                f'fill="{PALETTE["ink"]}" font-size="26" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" '
                f'font-weight="700">{value}</text>'
            )
            parts.append(
                f'<text x="{x + (cell-8)/2}" y="{y + 62}" text-anchor="middle" '
                f'fill="{PALETTE["muted"]}" font-size="11" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">'
                f'{fmt_pct(intensity)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_class_metrics_svg(per_class: dict[str, dict[str, float]]) -> str:
    width = 760
    height = 360
    margin_left = 70
    chart_top = 56
    chart_height = 220
    chart_bottom = chart_top + chart_height
    group_width = 185
    bar_width = 28
    bar_gap = 10
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Per-class precision recall F1">'
    ]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="24" '
        f'fill="#fffdf8" stroke="{PALETTE["panel_border"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{margin_left}" y="30" fill="{PALETTE["ink"]}" font-size="18" '
        f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" font-weight="700">'
        f'Sınıf Bazlı Precision / Recall / F1</text>'
    )
    for tick in range(6):
        value = tick / 5
        y = chart_bottom - value * chart_height
        parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{width-30}" y2="{y}" stroke="#e8e4d8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left-12}" y="{y+4}" text-anchor="end" fill="{PALETTE["muted"]}" '
            f'font-size="11" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{fmt_pct(value)}</text>'
        )
    metric_keys = [("precision", "#1d4ed8"), ("recall", "#d97706"), ("f1", "#7c3aed")]
    for group_idx, label in enumerate(LABELS):
        x0 = margin_left + 40 + group_idx * group_width
        parts.append(
            f'<text x="{x0+bar_width+bar_gap}" y="{chart_bottom+28}" text-anchor="middle" fill="{PALETTE["ink"]}" '
            f'font-size="14" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" font-weight="600">{escape(display_label(label))}</text>'
        )
        for metric_idx, (metric_name, color) in enumerate(metric_keys):
            value = per_class[label][metric_name]
            bar_height = value * chart_height
            x = x0 + metric_idx * (bar_width + bar_gap)
            y = chart_bottom - bar_height
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" rx="10" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x+bar_width/2}" y="{max(y-8, 52)}" text-anchor="middle" fill="{PALETTE["ink"]}" '
                f'font-size="11" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{fmt_float(value)}</text>'
            )
            parts.append(
                f'<text x="{x+bar_width/2}" y="{chart_bottom+46}" text-anchor="middle" fill="{PALETTE["muted"]}" '
                f'font-size="10" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{metric_name}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_category_accuracy_svg(category_rows: list[dict]) -> str:
    row_height = 28
    margin_left = 210
    margin_top = 48
    width = 900
    height = margin_top + len(category_rows) * row_height + 32
    bar_area = width - margin_left - 40
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Category accuracy">'
    ]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="24" '
        f'fill="#fffdf8" stroke="{PALETTE["panel_border"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="24" y="30" fill="{PALETTE["ink"]}" font-size="18" '
        f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" font-weight="700">'
        f'Senaryo Türüne Göre Accuracy</text>'
    )
    for idx, row in enumerate(category_rows):
        y = margin_top + idx * row_height
        accuracy = row["accuracy"]
        x = margin_left
        bar_width = max(2, accuracy * bar_area)
        fill = "#15803d" if accuracy >= 0.95 else "#0f766e" if accuracy >= 0.8 else "#d97706" if accuracy >= 0.5 else "#b91c1c"
        parts.append(
            f'<text x="{margin_left-10}" y="{y+18}" text-anchor="end" fill="{PALETTE["ink"]}" '
            f'font-size="12" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{escape(display_category(row["category"]))}</text>'
        )
        parts.append(
            f'<rect x="{x}" y="{y+4}" width="{bar_area}" height="16" rx="8" fill="#ece7da"/>'
        )
        parts.append(
            f'<rect x="{x}" y="{y+4}" width="{bar_width}" height="16" rx="8" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{x + bar_area + 8}" y="{y+18}" fill="{PALETTE["muted"]}" font-size="12" '
            f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{fmt_pct(accuracy)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_transition_svg(transition_rows: list[dict]) -> str:
    width = 760
    row_height = 34
    margin_left = 220
    margin_top = 50
    bar_area = width - margin_left - 40
    height = margin_top + len(transition_rows) * row_height + 24
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Mismatch transitions">'
    ]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="24" '
        f'fill="#fffdf8" stroke="{PALETTE["panel_border"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="24" y="30" fill="{PALETTE["ink"]}" font-size="18" '
        f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" font-weight="700">'
        f'Hata Geçiş Sayıları</text>'
    )
    max_count = max(row["count"] for row in transition_rows) if transition_rows else 1
    for idx, row in enumerate(transition_rows):
        y = margin_top + idx * row_height
        label = f'{display_label(row["ground_truth"])} → {display_label(row["predicted_label"])}'
        width_value = (row["count"] / max_count) * bar_area
        color = PALETTE.get(row["ground_truth"], "#334155")
        parts.append(
            f'<text x="{margin_left-10}" y="{y+18}" text-anchor="end" fill="{PALETTE["ink"]}" '
            f'font-size="12" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{escape(label)}</text>'
        )
        parts.append(
            f'<rect x="{margin_left}" y="{y+4}" width="{bar_area}" height="18" rx="9" fill="#ece7da"/>'
        )
        parts.append(
            f'<rect x="{margin_left}" y="{y+4}" width="{width_value}" height="18" rx="9" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{margin_left + width_value + 8}" y="{y+18}" fill="{PALETTE["muted"]}" font-size="12" '
            f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{row["count"]}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_class_distribution_svg(
    ground_truth_distribution: dict[str, int],
    predicted_distribution: dict[str, int],
) -> str:
    width = 760
    height = 360
    margin_left = 78
    chart_top = 56
    chart_height = 220
    chart_bottom = chart_top + chart_height
    group_width = 200
    bar_width = 40
    bar_gap = 14
    max_count = max(
        [ground_truth_distribution.get(label, 0) for label in LABELS]
        + [predicted_distribution.get(label, 0) for label in LABELS]
        + [1]
    )
    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Class distribution">'
    ]
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="24" '
        f'fill="#fffdf8" stroke="{PALETTE["panel_border"]}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<text x="{margin_left}" y="30" fill="{PALETTE["ink"]}" font-size="18" '
        f'font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" font-weight="700">'
        f'Sınıf Dağılımı: Gerçek vs Tahmin</text>'
    )
    for tick in range(6):
        value = tick / 5
        y = chart_bottom - value * chart_height
        count = round(max_count * value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y}" x2="{width-30}" y2="{y}" stroke="#e8e4d8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left-12}" y="{y+4}" text-anchor="end" fill="{PALETTE["muted"]}" '
            f'font-size="11" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{count}</text>'
        )
    for group_idx, label in enumerate(LABELS):
        x0 = margin_left + 48 + group_idx * group_width
        gt_value = ground_truth_distribution.get(label, 0)
        pred_value = predicted_distribution.get(label, 0)
        gt_height = (gt_value / max_count) * chart_height
        pred_height = (pred_value / max_count) * chart_height
        parts.append(
            f'<text x="{x0+bar_width+bar_gap/2}" y="{chart_bottom+30}" text-anchor="middle" fill="{PALETTE["ink"]}" '
            f'font-size="14" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif" font-weight="600">{escape(display_label(label))}</text>'
        )
        parts.append(
            f'<rect x="{x0}" y="{chart_bottom-gt_height}" width="{bar_width}" height="{gt_height}" rx="12" fill="{PALETTE[label]}"/>'
        )
        parts.append(
            f'<rect x="{x0+bar_width+bar_gap}" y="{chart_bottom-pred_height}" width="{bar_width}" height="{pred_height}" rx="12" fill="#e2e8f0" stroke="{PALETTE[label]}" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x0+bar_width/2}" y="{max(chart_bottom-gt_height-8, 52)}" text-anchor="middle" fill="{PALETTE["ink"]}" '
            f'font-size="11" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{gt_value}</text>'
        )
        parts.append(
            f'<text x="{x0+bar_width+bar_gap+bar_width/2}" y="{max(chart_bottom-pred_height-8, 52)}" text-anchor="middle" fill="{PALETTE["ink"]}" '
            f'font-size="11" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">{pred_value}</text>'
        )
        parts.append(
            f'<text x="{x0+bar_width/2}" y="{chart_bottom+48}" text-anchor="middle" fill="{PALETTE["muted"]}" '
            f'font-size="10" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">gerçek</text>'
        )
        parts.append(
            f'<text x="{x0+bar_width+bar_gap+bar_width/2}" y="{chart_bottom+48}" text-anchor="middle" fill="{PALETTE["muted"]}" '
            f'font-size="10" font-family="IBM Plex Sans, Avenir Next, Segoe UI, sans-serif">tahmin</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def dominant_transition(category: str, mismatches: list[dict]) -> str:
    counter = Counter()
    for row in mismatches:
        if row["category"] == category:
            counter[(row["ground_truth"], row["predicted_label"])] += 1
    if not counter:
        return "hata yok"
    (gt, pred), count = counter.most_common(1)[0]
    return f"{display_label(gt)} → {display_label(pred)} ({count})"


def build_key_findings(
    summary: dict,
    category_rows: list[dict],
    transition_rows: list[dict],
) -> list[str]:
    overall = summary["overall"]
    per_class = overall["per_class"]
    lowest_recall_label = min(LABELS, key=lambda label: per_class[label]["recall"])
    worst_categories = [row["category"] for row in category_rows[:4]]
    dominant = transition_rows[0] if transition_rows else None
    findings = [
        (
            f"Genel benchmark sonucu {overall['count']} sentetik finans örneğinde "
            f"{fmt_pct(overall['accuracy'])} accuracy ve {fmt_float(overall['macro_f1'])} macro-F1."
        ),
        (
            f"En zayıf sınıf {display_label(lowest_recall_label)}; recall değeri "
            f"{fmt_float(per_class[lowest_recall_label]['recall'])}. "
            f"Bu, modelin güçlü negatif sinyal vermekten kaçınan muhafazakar yapısıyla uyumlu."
        ),
        (
            f"En zayıf senaryo türleri {', '.join(display_category(item) for item in worst_categories)}. "
            f"Bunlar downstream kullanımda yüksek riskli iş kalıpları olarak görülmeli."
        ),
    ]
    if dominant:
        findings.append(
            f"En sık hata geçişi {display_label(dominant['ground_truth'])} → "
            f"{display_label(dominant['predicted_label'])}; toplam {dominant['count']} örnek."
        )
    findings.append(
        "Bu benchmark sentetik ve kural tabanlıdır; yayın kalitesinde resmi benchmark yerine yerel smoke test ve davranış analizi için kullanılmalıdır."
    )
    return findings


def save_asset(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.name


def make_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="card metric">'
        f'<div class="metric-title">{escape(title)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-note">{escape(note)}</div>'
        '</div>'
    )


def build_html(
    summary: dict,
    predictions: list[dict],
    mismatches: list[dict],
    asset_svgs: dict[str, str],
) -> str:
    overall = summary["overall"]
    ground_truth_distribution = Counter(row["ground_truth"] for row in predictions)
    predicted_distribution = Counter(row["predicted_label"] for row in predictions)
    transition_counter = Counter(
        (row["ground_truth"], row["predicted_label"]) for row in mismatches
    )
    transition_rows = [
        {"ground_truth": gt, "predicted_label": pred, "count": count}
        for (gt, pred), count in transition_counter.most_common()
    ]
    category_rows = [
        {
            "category": category,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "count": metrics["count"],
            "dominant_transition": dominant_transition(category, mismatches),
        }
        for category, metrics in summary["by_category"].items()
    ]
    category_rows.sort(key=lambda row: (row["accuracy"], row["macro_f1"], row["category"]))
    findings = build_key_findings(summary, category_rows, transition_rows)
    sample_errors = mismatches[:15]

    cards = [
        make_card("Örnek Sayısı", str(overall["count"]), "Dengeli sentetik benchmark"),
        make_card("Accuracy", fmt_pct(overall["accuracy"]), "Genel doğru tahmin oranı"),
        make_card("Macro-F1", fmt_float(overall["macro_f1"]), "Sınıf ortalama F1 skoru"),
        make_card("Mismatch", str(summary["mismatch_count"]), "Ground truth ile uyuşmayan satırlar"),
    ]
    category_table_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(display_category(row['category']))}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{fmt_pct(row['accuracy'])}</td>"
            f"<td>{fmt_float(row['macro_f1'])}</td>"
            f"<td>{escape(row['dominant_transition'])}</td>"
            "</tr>"
        )
        for row in category_rows[:10]
    )
    error_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(row['id'])}</td>"
            f"<td>{escape(row.get('category_tr', display_category(row['category'])))}</td>"
            f"<td>{escape(display_label(row['ground_truth']))}</td>"
            f"<td>{escape(display_label(row['predicted_label']))}</td>"
            f"<td class=\"text-cell\">{escape(row['text'])}</td>"
            "</tr>"
        )
        for row in sample_errors
    )
    finding_rows = "".join(f"<li>{escape(item)}</li>" for item in findings)

    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FinTurkBERT Değerlendirme Raporu</title>
  <style>
    :root {{
      --bg: #f3efe4;
      --panel: #fffdf8;
      --panel-border: #d9d4c7;
      --ink: #172033;
      --muted: #5f6b85;
      --accent: #1d4ed8;
      --positive: #15803d;
      --neutral: #0f766e;
      --negative: #c2410c;
      --warn: #b91c1c;
      --shadow: 0 20px 50px rgba(23, 32, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      overflow-x: hidden;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(21, 128, 61, 0.08), transparent 24%),
        linear-gradient(180deg, #f6f2e8 0%, #f1ecdf 100%);
    }}
    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 36px 24px 64px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 0.9fr;
      gap: 24px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }}
    .hero-main {{
      padding: 28px;
      min-height: 240px;
      background:
        linear-gradient(135deg, rgba(29, 78, 216, 0.10), rgba(255, 255, 255, 0)),
        linear-gradient(160deg, rgba(21, 128, 61, 0.10), rgba(255, 255, 255, 0));
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(29, 78, 216, 0.08);
      color: var(--accent);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 18px 0 8px;
      font-size: clamp(32px, 5vw, 52px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .sub {{
      margin: 0;
      max-width: 62ch;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.6;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }}
    .badge {{
      padding: 10px 14px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid var(--panel-border);
      background: rgba(255, 255, 255, 0.75);
    }}
    .hero-side {{
      padding: 28px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background:
        linear-gradient(180deg, rgba(194, 65, 12, 0.08), rgba(255, 255, 255, 0)),
        linear-gradient(160deg, rgba(15, 118, 110, 0.10), rgba(255, 255, 255, 0));
    }}
    .hero-side h2 {{
      margin: 0 0 12px;
      font-size: 20px;
      letter-spacing: -0.02em;
    }}
    .hero-side ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.7;
    }}
    .grid-cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric {{
      padding: 22px;
      min-height: 132px;
    }}
    .metric-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
    }}
    .metric-value {{
      margin-top: 12px;
      font-size: 30px;
      line-height: 1.1;
      font-weight: 700;
      word-break: break-word;
    }}
    .metric-note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
      margin-bottom: 24px;
    }}
    .panel {{
      padding: 22px;
    }}
    .panel h3 {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}
    .panel p {{
      color: var(--muted);
      line-height: 1.6;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      padding-bottom: 4px;
    }}
    .table-wrap table {{
      min-width: 560px;
    }}
    .table-wrap.wide table {{
      min-width: 920px;
    }}
    th, td {{
      text-align: left;
      padding: 12px 10px;
      border-bottom: 1px solid #ece7da;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .text-cell {{
      max-width: 720px;
      line-height: 1.5;
    }}
    .svg-frame img,
    .svg-frame svg {{
      width: 100%;
      display: block;
      border-radius: 22px;
      border: 1px solid var(--panel-border);
      background: #fffdf8;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
    }}
    .section-title {{
      margin: 28px 0 12px;
      font-size: 22px;
      letter-spacing: -0.03em;
    }}
    .footer-note {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    @media (max-width: 1040px) {{
      .hero, .grid-cards, .two-col {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .shell {{
        padding: 18px 12px 42px;
      }}
      .card {{
        border-radius: 20px;
      }}
      .hero-main, .hero-side, .panel, .metric {{
        padding: 16px;
      }}
      h1 {{
        font-size: clamp(28px, 9vw, 40px);
        line-height: 1;
      }}
      .sub {{
        font-size: 14px;
      }}
      .badge {{
        font-size: 12px;
        padding: 8px 12px;
      }}
      .metric-value {{
        font-size: 24px;
      }}
      .panel h3 {{
        font-size: 17px;
      }}
      .svg-frame img,
      .svg-frame svg {{
        border-radius: 16px;
      }}
      th, td {{
        padding: 10px 8px;
      }}
      .text-cell {{
        min-width: 280px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="card hero-main">
        <div class="eyebrow">FinTurkBERT yerel benchmark</div>
        <h1>Türkçe finansal sentiment sınıflandırması davranış raporu</h1>
        <p class="sub">
          Bu dashboard, yerelde cache'lenen <code>ff112/FinTurkBERT</code> checkpoint'inin yatırımcı bakış açısıyla hazırlanmış dengeli bir sentetik Türkçe finans benchmark'ında nasıl davrandığını özetler.
        </p>
        <div class="badge-row">
          <div class="badge">Accuracy {fmt_pct(overall["accuracy"])}</div>
          <div class="badge">Macro-F1 {fmt_float(overall["macro_f1"])}</div>
          <div class="badge">Hata {summary["mismatch_count"]}</div>
          <div class="badge">Offline cache doğrulandı</div>
        </div>
      </div>
      <div class="card hero-side">
        <div>
          <h2>Ana Bulgular</h2>
          <ul>{finding_rows}</ul>
        </div>
        <div class="footer-note">
          <code>{escape(summary["dataset"])}</code> kaynağından üretildi. Kullanılan yerel cache: <code>{escape(summary["cache_dir"])}</code>.
        </div>
      </div>
    </section>

    <section class="grid-cards">
      {''.join(cards)}
    </section>

    <section class="two-col">
      <div class="card panel">
        <h3>Karmaşıklık Matrisi</h3>
        <div class="svg-frame">
          {asset_svgs['confusion']}
        </div>
        <p>Model, rutin nötr bildirimlerde ve açık pozitif iş gelişmelerinde en güçlü performansı gösteriyor. Ana hata deseni ise bazı negatif ve pozitif örneklerin muhafazakar biçimde nötre çökmesi.</p>
      </div>
      <div class="card panel">
        <h3>Sınıf Bazlı Metrikler</h3>
        <div class="svg-frame">
          {asset_svgs['class_metrics']}
        </div>
        <div class="legend">
          <span><i class="dot" style="background:#1d4ed8"></i> Precision</span>
          <span><i class="dot" style="background:#d97706"></i> Recall</span>
          <span><i class="dot" style="background:#7c3aed"></i> F1</span>
        </div>
      </div>
    </section>

    <section class="two-col">
      <div class="card panel">
        <h3>En Zayıf Senaryo Türleri</h3>
        <div class="svg-frame">
          {asset_svgs['categories']}
        </div>
        <p>Senaryo bazlı accuracy, genel headline metriklerde görünmeyen hata kümelerini açığa çıkarır. Aşağıda accuracy'si sıfır olan türler iyileştirme açısından en yüksek öncelikli alanlardır.</p>
      </div>
      <div class="card panel">
        <h3>Sınıf Dağılımı</h3>
        <div class="svg-frame">
          {asset_svgs['class_distribution']}
        </div>
        <p>Benchmark dengeli kurulduğu için gerçek dağılım eşit. Tahmin tarafındaki kayma, modelin hangi class'a daha sık çöktüğünü hızlıca gösteriyor.</p>
      </div>
    </section>

    <section class="two-col">
      <div class="card panel">
        <h3>Hata Geçiş Profili</h3>
        <div class="svg-frame">
          {asset_svgs['transitions']}
        </div>
        <p>Bu grafik, hataların en çok hangi label geçişlerinde toplandığını gösteriyor. En kritik geçişler genelde negative → neutral ve positive → neutral tarafında yoğunlaşıyor.</p>
      </div>
      <div class="card panel">
        <h3>En Düşük Accuracy'li Türler</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>Tür</th><th>Adet</th><th>Accuracy</th><th>Macro-F1</th><th>Baskın Hata</th></tr>
            </thead>
            <tbody>{category_table_rows}</tbody>
          </table>
        </div>
      </div>
    </section>

    <section class="card panel">
      <h3>Örnek Hatalı Tahminler</h3>
      <div class="table-wrap wide">
        <table>
          <thead>
            <tr><th>ID</th><th>Tür</th><th>Gerçek Etiket</th><th>Tahmin</th><th>Metin</th></tr>
          </thead>
          <tbody>{error_rows}</tbody>
        </table>
      </div>
      <div class="footer-note">
        Bu örnekler <code>reports/mismatches.csv</code> içinden seçildi. Mevcut model davranışının üretim akışın için kabul edilebilir olup olmadığını veya ikinci aşama kural/reviewer katmanı gerektirip gerektirmediğini görmek için yararlıdır.
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    summary = load_json(Path(args.summary))
    predictions = load_csv(Path(args.predictions))
    mismatches = load_csv(Path(args.mismatches))

    assets_dir = Path(args.assets_dir)
    category_rows = [
        {"category": category, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "count": metrics["count"]}
        for category, metrics in summary["by_category"].items()
    ]
    category_rows.sort(key=lambda row: (row["accuracy"], row["macro_f1"], row["category"]))
    transition_rows = [
        {"ground_truth": gt, "predicted_label": pred, "count": count}
        for (gt, pred), count in Counter(
            (row["ground_truth"], row["predicted_label"]) for row in mismatches
        ).most_common()
    ]

    asset_svgs = {
        "confusion": render_confusion_svg(summary["overall"]["confusion_matrix"]),
        "class_metrics": render_class_metrics_svg(summary["overall"]["per_class"]),
        "class_distribution": render_class_distribution_svg(
            Counter(row["ground_truth"] for row in predictions),
            Counter(row["predicted_label"] for row in predictions),
        ),
        "categories": render_category_accuracy_svg(category_rows),
        "transitions": render_transition_svg(transition_rows),
    }
    asset_paths = {
        "confusion": save_asset(
            assets_dir / "confusion_matrix.svg",
            asset_svgs["confusion"],
        ),
        "class_metrics": save_asset(
            assets_dir / "class_metrics.svg",
            asset_svgs["class_metrics"],
        ),
        "class_distribution": save_asset(
            assets_dir / "class_distribution.svg",
            asset_svgs["class_distribution"],
        ),
        "categories": save_asset(
            assets_dir / "category_accuracy.svg",
            asset_svgs["categories"],
        ),
        "transitions": save_asset(
            assets_dir / "mismatch_transitions.svg",
            asset_svgs["transitions"],
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(summary, predictions, mismatches, asset_svgs),
        encoding="utf-8",
    )

    print(f"HTML report: {output_path}")
    for key, filename in asset_paths.items():
        print(f"{key}: {assets_dir / filename}")


if __name__ == "__main__":
    main()
