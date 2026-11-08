import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import prepare_corpus


class PreparationTests(unittest.TestCase):
    def test_curation_filters_boilerplate_and_duplicate_sentences(self):
        positive = 'Şirket için net kâr ilk çeyrek döneminde yüzde 25 arttı.'
        negative = 'Şirketin satış gelirleri yıllık yüzde 15 azaldı.'
        records = [{'id': i, 'content': text} for i, text in enumerate([
            positive, positive, negative, 'Burada yer alan bilgiler yatırım tavsiyesi değildir.'
        ])]
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / 'news.jsonl'
            source.write_text(''.join(json.dumps(r) + '\n' for r in records))
            args = prepare_corpus.build_cli().parse_args([
                'run-all', '--input', str(source), '--text-field', 'content',
                '--workdir', str(base / 'outputs'), '--lex-dir', str(base / 'lexicons'),
                '--total-target', '20', '--min-true', '1'
            ])
            prepare_corpus.run_all(args)
            rows = [json.loads(line) for line in (base / 'outputs/03_filtered.jsonl').read_text().splitlines()]
            self.assertEqual({r['sentence'] for r in rows}, {positive, negative})
            self.assertEqual(len(rows), 2)

    def test_notebook_syntax_and_packing(self):
        notebook = json.loads((ROOT / 'pretraining/domain_pretraining.ipynb').read_text())
        functions = []
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            self.assertEqual(cell['outputs'], [])
            if source.startswith('%'):
                continue
            tree = ast.parse(source)
            functions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'packing')

        namespace = {'PreTrainedTokenizer': object, 'tqdm': lambda rows, **kwargs: rows}
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<packing>', 'exec'), namespace)

        class Tokenizer:
            cls_token_id = 1
            sep_token_id = 2

            def encode(self, text, **kwargs):
                return [3] * len(text.split())

        blocks = namespace['packing']([[' '.join(['word'] * 600)]], Tokenizer())
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(len(block) <= 512 for block in blocks))
        self.assertTrue(all(block[0] == 1 and block[-1] == 2 for block in blocks))


if __name__ == '__main__':
    unittest.main()
