#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TR Finance Curation Pipeline (single-file demo)

This file merges THREE small, rule-based scripts into ONE coherent CLI tool:

1) Stage 1: "curate"
   - Read a large TR finance news corpus (JSONL or CSV)
   - Split into sentences, remove boilerplate
   - Keep only "finance_anchor AND trigger" sentences using lexicons
   - Bucketize and (optionally) deduplicate + reservoir sample per bucket
   - Output: JSONL of candidate sentences with feature flags

2) Stage 2: "downsample"
   - Take the Stage-1 candidates and compress them without an LLM:
     * template-normalize (mask numbers/currency/%/quarters/dates) and cap per template
     * bucket-wise reservoir sampling to hit a target total (balanced buckets)
   - Output: JSONL (e.g., 20k) suitable for annotation

3) Stage 3: "filter-features"
   - Keep only items whose feature flags contain at least N True values
   - Output: JSONL (e.g., more "high-signal" subset)

Why one file?
- Easy to demo on stage: `python tr_finance_pipeline_demo.py run-all ...`
- No imports across files; every stage is visible in one place.

---------------------------------------------------------------------------
Quick demo commands

# Stage 1 only:
python tr_finance_pipeline_demo.py curate \
  --input news.jsonl --output curated_candidates.jsonl --format jsonl \
  --lex-dir lex_tr_fin --use-stanza --use-morph --enable-simhash \
  --max-per-bucket 0

# Stage 2 only (downsample to 20k):
python tr_finance_pipeline_demo.py downsample \
  --input curated_candidates.jsonl --output curated_final_20000.jsonl \
  --total-target 20000 --max-per-template 3

# Stage 3 only (keep >=4 true feature flags):
python tr_finance_pipeline_demo.py filter-features \
  --input curated_final_20000.jsonl --output curated_final_20k_min4true.jsonl \
  --min-true 4

# Full pipeline in one shot (creates intermediate outputs):
python tr_finance_pipeline_demo.py run-all \
  --input news.jsonl --format jsonl --workdir ./out_demo \
  --total-target 20000 --max-per-template 3 --min-true 4 \
  --lex-dir lex_tr_fin --use-stanza --use-morph --enable-simhash

---------------------------------------------------------------------------
Notes
- "No LLM / no model": this pipeline is purely heuristic + lexicon based.
- If you enable Turkish morphology with --use-morph, it tries to use `zeyrek`
  (optional), otherwise falls back to a simple suffix-strip stemmer.
- If you enable --use-stanza, you need `pip install stanza` and the TR model:
    python -c "import stanza; stanza.download('tr')"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Optional progress bar (tqdm)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None, disable=False, **kwargs):
        if iterable is None:
            return range(total) if total else []
        return iterable

# =============================================================================
# Stage 1 — Lexicon-gated sentence curation
# =============================================================================

# -----------------------------
# Defaults (edit or override via --lex-dir)
# -----------------------------
DEFAULT_LEX = {
    "finance_terms.txt": [
        # core market / equity
        "bist", "endeks", "hisse", "pay", "lot", "borsa", "kapanış", "kapanis",
        "prim", "iskonto", "hacim", "işlem hacmi", "islem hacmi", "piyasa değeri", "piyasa degeri",
        # statements / disclosures
        "kap", "özel durum açıklaması", "ozel durum aciklamasi", "finansal tablo", "bilanço", "bilanco",
        "gelir tablosu", "faaliyet raporu",
        # financial metrics
        "ciro", "hasılat", "hasilat", "satış", "satis", "net kâr", "net kar", "zarar",
        "favök", "fvaök", "fvaok", "ebitda", "marj", "brüt kâr", "brut kar",
        "faaliyet kârı", "faaliyet kari", "net borç", "net borc", "nakit akışı", "nakit akisi",
        "özkaynak", "ozkaynak", "aktif toplam", "pasif", "likidite",
        # corporate actions
        "temettü", "temettu", "geri alım", "geri alim", "sermaye artırımı", "sermaye artirimi",
        "bedelli", "bedelsiz", "pay geri alım", "pay geri alim", "halka arz", "ikincil halka arz",
        # rates / macro frequently co-mentioned in finance news
        "faiz", "enflasyon", "kur", "dolar", "euro", "cds", "tahvil", "bono", "swap",
        # guidance/expectations
        "beklenti", "tahmin", "öngörü", "ongoru", "rehberlik", "hedef",
        # valuation / analyst
        "hedef fiyat", "tavsiye", "al", "tut", "sat", "rapor", "analist",
    ],
    "direction_up.txt": [
        "arttı", "artti", "yükseldi", "yukseldi", "çıktı", "cikti", "güçlendi", "guclendi",
        "büyüdü", "buyudu", "genişledi", "genisledi", "iyileşti", "iyilesti",
        "rekor", "zirve", "toparlandı", "toparlandi", "ivmelendi", "hızlandı", "hizlandi",
        "beklentiyi aştı", "beklentiyi asti", "beklentinin üzerinde", "beklentinin uzerinde",
        # variants
        "artış", "artışla", "artışının", "artarak", "iyileşme", "iyileştiğini",
        "yükseliş", "yükselerek", "yükselmiştir", "yükseltti", "toparlanma",
    ],
    "direction_down.txt": [
        "azaldı", "azaldi", "düştü", "dustu", "geriledi", "eksildi", "zayıfladı", "zayifladi",
        "bozuldu", "daraldı", "daraldi", "kayıp", "kayip", "satıcılı", "saticili",
        "beklentiyi kaçırdı", "beklentiyi kacirdi", "beklentinin altında", "beklentinin altinda",
        # variants
        "aşağı", "azalış", "azalarak", "düşüş", "düşerek", "gerileme", "kaybetti",
        "zorlu", "zorlaşan", "zorlayıcı",
    ],
    "risk_modality.txt": [
        "bekleniyor", "öngörülüyor", "ongoruluyor", "tahmin ediliyor", "muhtemelen",
        "olası", "olasi", "risk", "belirsizlik", "volatil", "baskı", "baski",
        "revize", "güncelledi", "guncelledi", "yönlendirme", "yonlendirme",
        # variants
        "beklenmedik", "endişe", "uyarısı",
    ],
    "legal_reg.txt": [
        "soruşturma", "sorusturma", "inceleme", "ceza", "dava", "yaptırım", "yaptirim",
        "iflas", "konkordato", "tasfiye", "spk", "rekabet kurumu", "mahkeme",
        "denetim", "usulsüz", "usulsuz", "suistimal", "manipülasyon", "manipulasyon",
    ],
    "ratings_actions.txt": [
        "not", "kredi notu", "rating", "görünüm", "gorunum",
        "yükseltti", "yukseltti", "düşürdü", "dusurdu", "upgrade", "downgrade",
        "hedef fiyat", "tavsiye", "al", "tut", "sat",
    ],
    "company_terms.txt": [
        # optional: BIST tickers or company names, one per line
        # Example: "THYAO", "ASELSAN", "Koç Holding"
    ],
    "boilerplate_contains.txt": [
        "yatırım tavsiyesi değildir", "yatirim tavsiyesi degildir",
        "burada yer alan", "sorumluluk kabul", "bilgilendirme amaçlı", "bilgilendirme amacli",
        "copyright", "tüm hakları saklıdır", "tum haklari saklidir",
        "çerez", "cerez", "cookie",
    ],
}

# -----------------------------
# Text utilities / regex
# -----------------------------
TR_LETTERS = "0-9A-Za-zğüşöçıİĞÜŞÖÇ"
WS_RE = re.compile(r"\s+")
SENT_SPLIT_FALLBACK = re.compile(r"(?<=[.!?…])\s+|\n+")

MONEY_NUM_RE = re.compile(
    r"(?i)(?:₺|\btl\b|\btry\b|\busd\b|\beur\b|\bgbp\b|%|\byüzde\b|\bmilyon\b|\bmilyar\b|\bbin\b)"
)
QUARTER_RE = re.compile(r"(?i)\b(çeyrek|ceyrek|q[1-4]|yıllık|yillik|aylık|aylik|yoy|qoq)\b")


def norm_text(s: str) -> str:
    s = s.casefold()
    s = WS_RE.sub(" ", s).strip()
    return s


def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def ensure_lex_dir(lex_dir: str) -> None:
    """
    Ensure lex_dir exists and has default lexicon files on first run.
    If files already exist, we DO NOT overwrite them.
    """
    os.makedirs(lex_dir, exist_ok=True)
    for fname, lines in DEFAULT_LEX.items():
        path = os.path.join(lex_dir, fname)
        if os.path.exists(path):
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write("# one entry per line; lines starting with # are ignored\n")
            for x in lines:
                f.write(x + "\n")


def load_lex(path: str) -> List[str]:
    """Load a lexicon file (one entry per line; # comment lines ignored)."""
    if not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    out.sort(key=len, reverse=True)  # longest first improves phrase matching
    return out


# -----------------------------
# Turkish morphological analysis (optional)
# -----------------------------
class TurkishMorphAnalyzer:
    """
    Handles Turkish postfix/suffix variants to improve lexicon matching.

    - If `zeyrek` is available, uses a morphological analyzer.
    - Otherwise falls back to a simple suffix-strip stemmer.

    This isn't perfect, but it improves recall substantially in practice.
    """
    def __init__(self, use_zeyrek: bool = True):
        self.use_zeyrek = use_zeyrek
        self._analyzer = None
        self._lock = threading.Lock()
        self._stem_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()

    def _init_zeyrek(self):
        if self._analyzer is not None:
            return
        with self._lock:
            if self._analyzer is not None:
                return
            try:
                from zeyrek import MorphAnalyzer  # type: ignore
                self._analyzer = MorphAnalyzer()
            except ImportError:
                self.use_zeyrek = False
                self._analyzer = None

    def _simple_stem(self, word: str) -> str:
        word = word.lower().strip()
        if len(word) < 3:
            return word

        # A very small suffix list; enough to catch common variants.
        suffixes = [
            "lerinden", "larindan", "lerinden",
            "lerine", "larina", "lerini", "larini",
            "lerinde", "larinda", "lerden", "lardan",
            "lerde", "larda", "lerin", "larin", "leri", "lari",
            "ler", "lar",
            "inden", "indan", "ine", "ina", "ini", "inde", "inda",
            "den", "dan", "de", "da",
            "in", "i", "ı", "u", "ü", "e", "a",
            "miş", "mis", "muş", "mus", "mış",
            "ti", "tı", "tu", "tü", "di", "dı", "du", "dü",
            "li", "lı", "lu", "lü",
            "ki",
        ]
        for suf in suffixes:
            if word.endswith(suf) and len(word) > len(suf) + 2:
                return word[:-len(suf)]
        return word

    def get_stem(self, word: str) -> str:
        word_lower = word.lower().strip()

        with self._cache_lock:
            if word_lower in self._stem_cache:
                return self._stem_cache[word_lower]

        stem = None
        if self.use_zeyrek:
            try:
                self._init_zeyrek()
                if self._analyzer is not None:
                    analyses = self._analyzer.analyze(word_lower)
                    if analyses:
                        first = analyses[0]
                        if isinstance(first, dict):
                            stem = first.get("root") or first.get("lemma") or first.get("stem")
                        elif hasattr(first, "root"):
                            stem = first.root
                        elif hasattr(first, "lemma"):
                            stem = first.lemma
                        elif hasattr(first, "stem"):
                            stem = first.stem
            except Exception:
                stem = None

        if not stem:
            stem = self._simple_stem(word_lower)

        with self._cache_lock:
            self._stem_cache[word_lower] = stem
        return stem

    def normalize_word(self, word: str) -> str:
        word = word.lower().strip()
        return word.replace("ı", "i").replace("İ", "i")


def build_phrase_regex(phrases: List[str], morph: Optional[TurkishMorphAnalyzer] = None) -> Optional[re.Pattern]:
    """
    Compile a regex that matches any phrase in `phrases` with "soft boundaries".

    If `morph` is provided, we add stems/normalized forms for single-word phrases
    to catch more Turkish morphological variants.
    """
    if not phrases:
        return None

    all_phrases: Set[str] = set()
    for phrase in phrases:
        pl = phrase.lower()
        all_phrases.add(pl)

        if morph and " " not in phrase.strip():
            stem = morph.get_stem(phrase)
            if stem and stem != pl:
                all_phrases.add(stem)
            all_phrases.add(morph.normalize_word(phrase))

    if not all_phrases:
        return None

    esc = [re.escape(p) for p in sorted(all_phrases, key=len, reverse=True)]
    pat = r"(?i)(?:^|[^" + TR_LETTERS + r"])(?:" + "|".join(esc) + r")(?:[^" + TR_LETTERS + r"]|$)"
    return re.compile(pat)


def sent_split_fallback(text: str) -> List[str]:
    """Lightweight sentence splitting if stanza isn't available/desired."""
    text = text.replace("\r", "\n")
    text = WS_RE.sub(" ", text).strip()
    if not text:
        return []
    parts = SENT_SPLIT_FALLBACK.split(text)
    return [p.strip() for p in parts if p and len(p.strip()) > 2]


class SentenceSplitter:
    """Optional stanza-based TR sentence splitter (higher quality)."""
    def __init__(self, use_stanza: bool):
        self.use_stanza = use_stanza
        self._nlp = None
        self._lock = threading.Lock()

    def _init(self):
        if self._nlp is not None:
            return
        with self._lock:
            if self._nlp is not None:
                return
            import stanza  # type: ignore
            self._nlp = stanza.Pipeline(
                lang="tr",
                processors="tokenize",
                tokenize_no_ssplit=False,
                verbose=False
            )

    def split(self, text: str) -> List[str]:
        if not self.use_stanza:
            return sent_split_fallback(text)
        try:
            self._init()
            doc = self._nlp(text)  # type: ignore
            sents = [s.text.strip() for s in doc.sentences if s.text and len(s.text.strip()) > 2]
            return sents if sents else sent_split_fallback(text)
        except Exception:
            return sent_split_fallback(text)


# -----------------------------
# Reservoir sampling per bucket
# -----------------------------
@dataclass
class Reservoir:
    k: int
    n_seen: int = 0
    items: List[dict] = None
    _lock: threading.Lock = None

    def __post_init__(self):
        if self.items is None:
            self.items = []
        if self._lock is None:
            self._lock = threading.Lock()

    def consider(self, item: dict, rng: random.Random) -> None:
        with self._lock:
            self.n_seen += 1
            if self.k <= 0:
                return
            if len(self.items) < self.k:
                self.items.append(item)
                return
            j = rng.randrange(self.n_seen)
            if j < self.k:
                self.items[j] = item


# -----------------------------
# SimHash (optional near-dup)
# -----------------------------
def _tokenize_for_simhash(s: str) -> List[str]:
    s = norm_text(s)
    s = re.sub(r"[^0-9a-zğüşöçı]+", " ", s, flags=re.IGNORECASE)
    return [t for t in s.split() if len(t) >= 2]


def simhash64(text: str) -> int:
    toks = _tokenize_for_simhash(text)
    if not toks:
        return 0
    v = [0] * 64
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        x = (h ^ (h >> 64)) & ((1 << 64) - 1)  # fold md5 -> 64-bit
        for i in range(64):
            v[i] += 1 if ((x >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class SimHashDeduper:
    """
    Near-duplicate check via SimHash:
    - bucket hashes into bands by prefix bits
    - compare only inside same band
    """
    def __init__(self, threshold: int = 3, band_bits: int = 16, max_bucket_list: int = 50000):
        self.threshold = threshold
        self.band_bits = band_bits
        self.max_bucket_list = max_bucket_list
        self.buckets: Dict[int, List[int]] = {}
        self._lock = threading.Lock()

    def _key(self, h: int) -> int:
        return h >> (64 - self.band_bits)

    def is_near_dup(self, h: int) -> bool:
        k = self._key(h)
        with self._lock:
            for prev in self.buckets.get(k, []):
                if hamming64(h, prev) <= self.threshold:
                    return True
        return False

    def add(self, h: int) -> None:
        k = self._key(h)
        with self._lock:
            lst = self.buckets.setdefault(k, [])
            if len(lst) < self.max_bucket_list:
                lst.append(h)


# -----------------------------
# Feature gating + bucketization
# -----------------------------
def is_boilerplate(sentence: str, boiler_subs: List[str]) -> bool:
    ns = norm_text(sentence)
    return any(sub and sub.casefold() in ns for sub in boiler_subs)


def bucketize(fin: bool, comp: bool, up: bool, down: bool, legal: bool, modal: bool, rating: bool, has_num: bool) -> str:
    # prioritize more semantically meaningful buckets first
    if legal:
        return "legal_reg"
    if rating:
        return "rating_action"
    if up and not down:
        return "finance_up"
    if down and not up:
        return "finance_down"
    if up and down:
        return "finance_updown"
    if modal:
        return "guidance_modal"
    if has_num:
        return "numeric_finance"
    if fin or comp:
        return "finance_other"
    return "other"


class SharedState:
    """Thread-safe shared state across Stage-1 worker threads."""
    def __init__(self, args, regexes, splitter, simdedup):
        self.args = args
        self.regexes = regexes
        self.splitter = splitter
        self.simdedup = simdedup
        self.seen_exact: Set[str] = set()
        self.seen_lock = threading.Lock()
        self.reservoirs: Dict[str, Reservoir] = {}
        self.reservoir_lock = threading.Lock()
        self.rng = random.Random(args.seed)

    def get_reservoir(self, bucket: str) -> Reservoir:
        with self.reservoir_lock:
            if bucket not in self.reservoirs:
                self.reservoirs[bucket] = Reservoir(k=self.args.max_per_bucket)
            return self.reservoirs[bucket]

    def is_seen_exact(self, hx: str) -> bool:
        with self.seen_lock:
            return hx in self.seen_exact

    def add_seen_exact(self, hx: str) -> None:
        with self.seen_lock:
            self.seen_exact.add(hx)


def process_record_stage1(
    obj: dict,
    state: SharedState,
    boiler_subs: List[str],
    text_field: str,
    id_field: str,
    date_field: str,
    source_field: str,
) -> List[dict]:
    """
    Stage-1 worker: takes a doc record and returns curated sentence items.
    """
    text = obj.get(text_field, "")
    if not isinstance(text, str) or not text.strip():
        return []

    doc_id = obj.get(id_field, None)
    date_val = obj.get(date_field, None) if date_field else None
    source_val = obj.get(source_field, None) if source_field else None

    sents = state.splitter.split(text)
    if not sents:
        return []

    re_fin, re_comp, re_up, re_down, re_risk, re_legal, re_rating = state.regexes

    def m(pat: Optional[re.Pattern], s: str) -> bool:
        return bool(pat and pat.search(" " + s + " "))

    def keep_rule(fin_or_comp: bool, up: bool, down: bool, legal: bool, modal: bool, rating: bool, has_num: bool) -> bool:
        # "finance anchor" AND at least one "trigger"
        trigger = up or down or legal or modal or rating or has_num
        return fin_or_comp and trigger

    items_out: List[dict] = []

    for i, sent in enumerate(sents):
        sent = sent.strip()
        if len(sent) < state.args.min_chars or len(sent) > state.args.max_chars:
            continue
        if is_boilerplate(sent, boiler_subs):
            continue

        # exact dedup (global)
        ne = norm_text(sent)
        hx = md5_hex(ne)
        if state.is_seen_exact(hx):
            continue

        # optional near-dup (SimHash)
        if state.simdedup is not None:
            sh = simhash64(sent)
            if state.simdedup.is_near_dup(sh):
                continue
            state.simdedup.add(sh)

        fin = m(re_fin, sent)
        comp = m(re_comp, sent)
        fin_or_comp = fin or comp

        up = m(re_up, sent)
        down = m(re_down, sent)
        legal = m(re_legal, sent)
        modal = m(re_risk, sent) or bool(QUARTER_RE.search(sent))
        rating = m(re_rating, sent)
        has_num = bool(MONEY_NUM_RE.search(sent))

        if not keep_rule(fin_or_comp, up, down, legal, modal, rating, has_num):
            continue

        b = bucketize(fin, comp, up, down, legal, modal, rating, has_num)

        item = {
            "doc_id": doc_id,
            "date": date_val,
            "source": source_val,
            "sentence": sent,
            "bucket": b,
            "features": {
                "finance_term": fin,
                "company_term": comp,
                "direction_up": up,
                "direction_down": down,
                "legal_reg": legal,
                "guidance_modal": modal,
                "rating_action": rating,
                "money_or_number": has_num,
            },
        }

        if state.args.keep_context_window == 1:
            prev_s = sents[i - 1].strip() if i - 1 >= 0 else ""
            next_s = sents[i + 1].strip() if i + 1 < len(sents) else ""
            item["context"] = {"prev": prev_s, "next": next_s}

        state.add_seen_exact(hx)

        # Reservoir sampling per bucket, or emit immediately if max_per_bucket==0
        if state.args.max_per_bucket > 0:
            state.get_reservoir(b).consider(item, state.rng)
        else:
            items_out.append(item)

    return items_out


def _count_lines_fast(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def iter_records(input_path: str, fmt: str) -> Tuple[Iterator[dict], Optional[int]]:
    """
    Streaming record iterator + optional total size for progress bars.
    """
    if fmt == "jsonl":
        total = _count_lines_fast(input_path)
        def gen() -> Iterator[dict]:
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue
        return gen(), total

    # csv
    # For CSV, count lines (minus header) as a rough total.
    total = _count_lines_fast(input_path)
    if total is not None:
        total = max(0, total - 1)

    def gen_csv() -> Iterator[dict]:
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
    return gen_csv(), total


def run_stage1_curate(args: argparse.Namespace) -> int:
    """
    Stage 1 entry point: write curated candidate JSONL.
    Returns number of items written.
    """
    ensure_lex_dir(args.lex_dir)

    finance_terms = load_lex(os.path.join(args.lex_dir, "finance_terms.txt"))
    company_terms = load_lex(os.path.join(args.lex_dir, "company_terms.txt"))
    up_terms = load_lex(os.path.join(args.lex_dir, "direction_up.txt"))
    down_terms = load_lex(os.path.join(args.lex_dir, "direction_down.txt"))
    risk_terms = load_lex(os.path.join(args.lex_dir, "risk_modality.txt"))
    legal_terms = load_lex(os.path.join(args.lex_dir, "legal_reg.txt"))
    rating_terms = load_lex(os.path.join(args.lex_dir, "ratings_actions.txt"))
    boiler_subs = load_lex(os.path.join(args.lex_dir, "boilerplate_contains.txt"))

    morph = TurkishMorphAnalyzer(use_zeyrek=True) if args.use_morph else None
    if args.use_morph:
        print("[curate] Turkish morphological analysis enabled.", file=sys.stderr)

    regexes = (
        build_phrase_regex(finance_terms, morph),
        build_phrase_regex(company_terms, morph),
        build_phrase_regex(up_terms, morph),
        build_phrase_regex(down_terms, morph),
        build_phrase_regex(risk_terms, morph),
        build_phrase_regex(legal_terms, morph),
        build_phrase_regex(rating_terms, morph),
    )

    splitter = SentenceSplitter(use_stanza=args.use_stanza)

    simdedup = None
    if args.enable_simhash:
        simdedup = SimHashDeduper(threshold=args.simhash_threshold, band_bits=args.simhash_band_bits)

    state = SharedState(args=args, regexes=regexes, splitter=splitter, simdedup=simdedup)

    records_it, total = iter_records(args.input, args.format)
    num_workers = args.workers if args.workers is not None else (os.cpu_count() or 4)

    written = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ThreadPool processing, streaming
    with open(out_path, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            # executor.map keeps memory low; results are returned in order
            fn = lambda obj: process_record_stage1(
                obj,
                state,
                boiler_subs,
                args.text_field,
                args.id_field,
                args.date_field,
                args.source_field,
            )
            for items in tqdm(ex.map(fn, records_it, chunksize=10), total=total, desc="Curating", unit="docs"):
                for item in items:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written += 1

        # If using per-bucket reservoirs, flush them at the end
        if args.max_per_bucket > 0:
            flushed = 0
            for res in state.reservoirs.values():
                for item in res.items:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    flushed += 1
            written += flushed
            print(f"[curate] Flushed {flushed} items from reservoirs.", file=sys.stderr)

    print(f"[curate] Done. Wrote {written} items -> {out_path}", file=sys.stderr)
    return written


# =============================================================================
# Stage 2 — Downsample candidates to a target size (no LLM)
# =============================================================================

# Bucket balance defaults (tune freely)
BUCKET_FRACTIONS = {
    "finance_up": 0.20,
    "finance_down": 0.20,
    "legal_reg": 0.15,
    "guidance_modal": 0.15,
    "rating_action": 0.10,
    "numeric_finance": 0.10,
    "finance_other": 0.10,
    "finance_updown": 0.00,
}

# Template normalization regex
RE_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
RE_CUR = re.compile(r"(?i)(₺|\btl\b|\btry\b|\busd\b|\beur\b|\bgbp\b|\bdolar\b|\beuro\b)")
RE_PCT = re.compile(r"(?i)(%|\byüzde\b)")
RE_QTR = re.compile(r"(?i)\b(q[1-4]|[1-4]\.?\s*çeyrek|çeyrek|ceyrek|yoy|qoq)\b")
RE_DATE = re.compile(r"(?i)\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4})\b")


def templ(s: str) -> str:
    """Normalize a sentence into a "template" to cap repetitive patterns."""
    s = s.casefold()
    s = RE_DATE.sub(" <DATE> ", s)
    s = RE_QTR.sub(" <QTR> ", s)
    s = RE_PCT.sub(" <PCT> ", s)
    s = RE_CUR.sub(" <CUR> ", s)
    s = RE_NUM.sub(" <NUM> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class ReservoirSimple:
    """Simple reservoir sampler used in stage-2."""
    def __init__(self, k: int, seed: int):
        self.k = k
        self.n = 0
        self.items: List[dict] = []
        self.rng = random.Random(seed)

    def consider(self, item: dict) -> None:
        if self.k <= 0:
            return
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return
        j = self.rng.randrange(self.n)
        if j < self.k:
            self.items[j] = item


def run_stage2_downsample(args: argparse.Namespace) -> int:
    """
    Stage 2 entry point: read stage-1 candidates and produce a balanced, smaller set.
    Returns number of items written.
    """
    rng = random.Random(args.seed)

    # Convert fractions -> target counts
    pos = {b: f for b, f in BUCKET_FRACTIONS.items() if f > 0}
    s = sum(pos.values())
    if s <= 0:
        raise ValueError("BUCKET_FRACTIONS sum must be > 0")

    bucket_targets = {b: int(round(args.total_target * (f / s))) for b, f in pos.items()}

    # Fix rounding to hit total exactly
    diff = args.total_target - sum(bucket_targets.values())
    if diff != 0:
        order = sorted(bucket_targets.items(), key=lambda x: -x[1])
        i = 0
        while diff != 0 and order:
            b = order[i % len(order)][0]
            step = 1 if diff > 0 else -1
            bucket_targets[b] = max(0, bucket_targets[b] + step)
            diff -= step
            i += 1

    reservoirs = {b: ReservoirSimple(k, seed=args.seed + idx) for idx, (b, k) in enumerate(bucket_targets.items())}
    template_counts: Dict[str, int] = defaultdict(int)

    seen_buckets = defaultdict(int)
    kept_after_template = defaultdict(int)

    total_lines = _count_lines_fast(args.input)

    with open(args.input, "r", encoding="utf-8") as f:
        pbar = tqdm(f, total=total_lines, desc="Downsampling", unit="lines")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            b = obj.get("bucket", None)
            if b not in reservoirs:
                continue

            sent = obj.get("sentence", "")
            if not isinstance(sent, str) or not sent.strip():
                continue

            seen_buckets[b] += 1

            t = templ(sent)
            if template_counts[t] >= args.max_per_template:
                continue
            template_counts[t] += 1
            kept_after_template[b] += 1

            reservoirs[b].consider(obj)

            pbar.set_postfix({"kept": sum(kept_after_template.values()), "templates": len(template_counts)})
        pbar.close()

    # Collect + shuffle final set
    out: List[dict] = []
    for b in reservoirs:
        out.extend(reservoirs[b].items)
    rng.shuffle(out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as w:
        for obj in tqdm(out, desc="Writing", unit="items"):
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("[downsample] DONE", file=sys.stderr)
    print("[downsample] Input:", args.input, file=sys.stderr)
    print("[downsample] Output:", str(out_path), file=sys.stderr)
    print("[downsample] Total written:", len(out), file=sys.stderr)
    print("[downsample] Bucket targets:", bucket_targets, file=sys.stderr)
    print("[downsample] Unique templates:", len(template_counts), file=sys.stderr)

    return len(out)


# =============================================================================
# Stage 3 — Filter by min-true feature flags
# =============================================================================

def count_true_features(feat: dict) -> int:
    if not isinstance(feat, dict):
        return 0
    return sum(1 for v in feat.values() if v is True)


def run_stage3_filter_features(args: argparse.Namespace) -> int:
    """
    Stage 3 entry point: keep only items with >= N true feature flags.
    Returns number of items written.
    """
    total_lines = _count_lines_fast(args.input)

    kept = 0
    total = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
        pbar = tqdm(f, total=total_lines, desc="Filtering", unit="lines")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            feats = obj.get("features", {})
            n_true = count_true_features(feats)
            if n_true >= args.min_true:
                obj["true_feature_count"] = n_true
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            pbar.set_postfix({"kept": kept, "read": total})
        pbar.close()

    print(f"[filter-features] Read: {total} | Kept (>= {args.min_true}): {kept} -> {out_path}", file=sys.stderr)
    return kept


# =============================================================================
# Orchestration — run-all
# =============================================================================

def run_all(args: argparse.Namespace) -> None:
    """
    Convenience wrapper: Stage1 -> Stage2 -> Stage3.
    Writes:
      workdir/01_candidates.jsonl
      workdir/02_downsampled.jsonl
      workdir/03_filtered.jsonl
    """
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    candidates = workdir / "01_candidates.jsonl"
    downsampled = workdir / "02_downsampled.jsonl"
    filtered = workdir / "03_filtered.jsonl"

    # Stage 1
    s1 = argparse.Namespace(**vars(args))
    s1.output = str(candidates)
    run_stage1_curate(s1)

    # Stage 2
    s2 = argparse.Namespace(**vars(args))
    s2.input = str(candidates)
    s2.output = str(downsampled)
    run_stage2_downsample(s2)

    # Stage 3 (optional)
    if args.min_true is not None and args.min_true > 0:
        s3 = argparse.Namespace(**vars(args))
        s3.input = str(downsampled)
        s3.output = str(filtered)
        run_stage3_filter_features(s3)

        print(f"[run-all] FINAL (filtered) -> {filtered}", file=sys.stderr)
    else:
        print(f"[run-all] FINAL (downsampled) -> {downsampled}", file=sys.stderr)


# =============================================================================
# CLI
# =============================================================================

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tr_finance_pipeline_demo.py",
        description="Single-file demo: TR finance curation -> downsample -> feature-filter (no LLM)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # -------- Stage 1: curate --------
    p1 = sub.add_parser("curate", help="Stage 1: lexicon-gated sentence curation")
    p1.add_argument("--input", required=True)
    p1.add_argument("--output", required=True)
    p1.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    p1.add_argument("--text-field", default="text")
    p1.add_argument("--id-field", default="id")
    p1.add_argument("--date-field", default="")
    p1.add_argument("--source-field", default="")
    p1.add_argument("--lex-dir", default="lex_tr_fin")

    p1.add_argument("--use-stanza", action="store_true")
    p1.add_argument("--use-morph", action="store_true", help="Enable Turkish morphology (zeyrek if available)")

    p1.add_argument("--min-chars", type=int, default=20)
    p1.add_argument("--max-chars", type=int, default=400)
    p1.add_argument("--keep-context-window", type=int, default=0, help="0=off, 1=prev/next included")

    p1.add_argument("--max-per-bucket", type=int, default=0, help="Reservoir cap per bucket; 0=no cap")
    p1.add_argument("--seed", type=int, default=42)

    p1.add_argument("--enable-simhash", action="store_true")
    p1.add_argument("--simhash-threshold", type=int, default=3)
    p1.add_argument("--simhash-band-bits", type=int, default=16)

    p1.add_argument("--workers", type=int, default=None, help="Thread workers (default: CPU count)")
    p1.set_defaults(_fn=run_stage1_curate)

    # -------- Stage 2: downsample --------
    p2 = sub.add_parser("downsample", help="Stage 2: template cap + bucket reservoir downsample")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output", required=True)
    p2.add_argument("--total-target", type=int, default=20000)
    p2.add_argument("--max-per-template", type=int, default=3)
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(_fn=run_stage2_downsample)

    # -------- Stage 3: filter-features --------
    p3 = sub.add_parser("filter-features", help="Stage 3: keep only items with >=N true features")
    p3.add_argument("--input", required=True)
    p3.add_argument("--output", required=True)
    p3.add_argument("--min-true", type=int, default=4)
    p3.set_defaults(_fn=run_stage3_filter_features)

    # -------- run-all --------
    p4 = sub.add_parser("run-all", help="Run Stage1 -> Stage2 -> Stage3 into a workdir")
    p4.add_argument("--input", required=True)
    p4.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    p4.add_argument("--text-field", default="text")
    p4.add_argument("--id-field", default="id")
    p4.add_argument("--date-field", default="")
    p4.add_argument("--source-field", default="")
    p4.add_argument("--lex-dir", default="lex_tr_fin")
    p4.add_argument("--use-stanza", action="store_true")
    p4.add_argument("--use-morph", action="store_true")
    p4.add_argument("--min-chars", type=int, default=20)
    p4.add_argument("--max-chars", type=int, default=400)
    p4.add_argument("--keep-context-window", type=int, default=0)
    p4.add_argument("--max-per-bucket", type=int, default=0)
    p4.add_argument("--enable-simhash", action="store_true")
    p4.add_argument("--simhash-threshold", type=int, default=3)
    p4.add_argument("--simhash-band-bits", type=int, default=16)
    p4.add_argument("--workers", type=int, default=None)

    p4.add_argument("--total-target", type=int, default=20000)
    p4.add_argument("--max-per-template", type=int, default=3)

    # If min_true <=0, we skip stage 3.
    p4.add_argument("--min-true", type=int, default=4)

    p4.add_argument("--seed", type=int, default=42)
    p4.add_argument("--workdir", required=True, help="Directory for intermediate outputs")

    p4.set_defaults(_fn=None)

    return ap


def main() -> None:
    ap = build_cli()
    args = ap.parse_args()

    if args.cmd == "run-all":
        run_all(args)
        return

    # other stages
    fn = getattr(args, "_fn", None)
    if fn is None:
        ap.error("No function registered for this command.")
    fn(args)


if __name__ == "__main__":
    main()
