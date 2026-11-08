import csv
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import build_html_report as report
import evaluate_finturkbert as evaluation
import generate_benchmark_dataset as benchmark
import generate_phrasebank_synthetic as phrasebank


class EvaluationTests(unittest.TestCase):
    def test_generators_are_deterministic_and_unique(self):
        for module in (benchmark, phrasebank):
            rows = module.generate_rows(2)
            self.assertEqual(rows, module.generate_rows(2))
            self.assertEqual(len(rows), len(module.FAMILIES) * 2)
            self.assertEqual(len({r['text'] for r in rows}), len(rows))
            self.assertEqual({r['ground_truth'] for r in rows}, set(evaluation.LABELS))

    def test_macro_f1_counts_predicted_only_class(self):
        result = evaluation.compute_metrics([
            {'ground_truth': 'positive', 'predicted_label': 'positive'},
            {'ground_truth': 'positive', 'predicted_label': 'negative'},
        ])
        self.assertEqual(result['accuracy'], 0.5)
        self.assertAlmostEqual(result['macro_f1'], 2 / 9, places=4)

    def test_confusion_percentages_are_row_normalized(self):
        confusion = {a: {b: 0 for b in evaluation.LABELS} for a in evaluation.LABELS}
        confusion['negative']['negative'] = 1
        confusion['negative']['neutral'] = 1
        confusion['positive']['positive'] = 10
        svg = report.render_confusion_svg(confusion)
        ET.fromstring(svg)
        self.assertEqual(svg.count('50.0%'), 2)

    def test_svg_assets_are_valid_xml(self):
        counts = {label: 1 for label in evaluation.LABELS}
        ET.fromstring(report.render_class_distribution_svg(counts, counts))
        ET.fromstring(report.render_transition_svg([]))

    def test_perfect_run_clears_previous_mismatches_and_builds_report(self):
        rows = benchmark.generate_rows(1)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            dataset = base / 'dataset.jsonl'
            benchmark.write_jsonl(dataset, rows)
            summary = base / 'summary.json'
            predictions = base / 'predictions.csv'
            mismatches = base / 'mismatches.csv'
            args = ['evaluate', '--dataset', str(dataset), '--summary-out', str(summary),
                    '--predictions-out', str(predictions), '--mismatches-out', str(mismatches)]

            def predict(texts, tokenizer, model, **kwargs):
                return [{'label': r['ground_truth'], 'scores': {
                    label: float(label == r['ground_truth']) for label in evaluation.LABELS
                }} for r in rows]

            mismatches.write_text('old,incorrect,output\n')
            with patch.object(sys, 'argv', args), patch.object(evaluation, 'load_model', return_value=(None, None)), patch.object(evaluation, 'predict_texts', side_effect=predict):
                evaluation.main()
            with mismatches.open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertEqual(json.loads(summary.read_text())['overall']['accuracy'], 1.0)
            html = base / 'report.html'
            args = ['report', '--summary', str(summary), '--predictions', str(predictions),
                    '--mismatches', str(mismatches), '--output', str(html), '--assets-dir', str(base / 'assets')]
            with patch.object(sys, 'argv', args):
                report.main()
            self.assertTrue(html.exists())
            self.assertNotIn('Offline cache doğrulandı', html.read_text())
            for asset in (base / 'assets').glob('*.svg'):
                ET.parse(asset)


if __name__ == '__main__':
    unittest.main()
