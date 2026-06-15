#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path

from finturkbert_utils import tr_category, tr_difficulty, tr_label


SEED = 20260329
COMPANIES = [
    "Aksa Enerji",
    "Anatolia Gıda",
    "Birlik Çelik",
    "Bora Lojistik",
    "Delta Yatırım Holding",
    "Eksen Teknoloji",
    "Formel Kimya",
    "Güney Perakende",
    "Hedef Savunma",
    "Işık Sigorta",
    "Kuzey Madencilik",
    "Lima Gayrimenkul",
    "Mavi Makine",
    "Nova Sağlık",
    "Orion Otomotiv",
    "Pera Yazılım",
    "Rota Cam",
    "Sera Ambalaj",
    "Trend Tarım",
    "Vizyon Telekom",
]
PERIODS = [
    "2025 ilk çeyrek",
    "2025 ikinci çeyrek",
    "2025 üçüncü çeyrek",
    "2025 yıl sonu",
    "2026 ilk çeyrek",
    "2026 ilk yarı",
]
COUNTRIES = [
    "Almanya",
    "Birleşik Arap Emirlikleri",
    "İtalya",
    "İspanya",
    "Katar",
    "Mısır",
    "Polonya",
    "Romanya",
    "Suudi Arabistan",
]
CITIES = [
    "Ankara",
    "Bursa",
    "Gaziantep",
    "İzmir",
    "Kocaeli",
    "Konya",
    "Manisa",
    "Samsun",
    "Tekirdağ",
]
PEOPLE = [
    "Ayşe Demir",
    "Barış Yalçın",
    "Cem Arslan",
    "Deniz Uslu",
    "Elif Korkmaz",
    "Mert Sevinç",
    "Nazlı Acar",
    "Onur Şahin",
    "Selin Aksoy",
    "Tolga Eren",
]
DATES = [
    "7 Nisan 2026",
    "12 Nisan 2026",
    "18 Nisan 2026",
    "24 Nisan 2026",
    "6 Mayıs 2026",
    "14 Mayıs 2026",
    "28 Mayıs 2026",
]

FAMILIES = [
    {
        "label": "positive",
        "difficulty": "clear",
        "category": "earnings_upgrade",
        "template": (
            "{company}, {period} döneminde net kârını %{profit_growth} artırarak "
            "{profit_m} milyon TL'ye çıkardı; yönetim yıl sonu gelir beklentisini "
            "%{guide_up} yukarı revize etti."
        ),
        "rationale": (
            "Net kâr artışı ve yukarı yönlü beklenti güncellemesi yatırımcı açısından "
            "açık biçimde olumludur."
        ),
    },
    {
        "label": "positive",
        "difficulty": "clear",
        "category": "contract_win",
        "template": (
            "{company}, {country} pazarında {contract_m} milyon TL tutarında "
            "{duration_year} yıllık satış sözleşmesi imzaladı; şirket anlaşmanın "
            "önümüzdeki 12 ay gelirlerine doğrudan katkı sağlayacağını belirtti."
        ),
        "rationale": (
            "Yeni ve parasal büyüklüğü açık sözleşme, gelecekteki gelir akışı için "
            "somut bir pozitif sinyaldir."
        ),
    },
    {
        "label": "positive",
        "difficulty": "clear",
        "category": "deleveraging",
        "template": (
            "{company}, varlık satışı sonrasında kısa vadeli finansal borcunu "
            "{debt_reduction_m} milyon TL azalttı; net borç/FAVÖK oranı "
            "{leverage_down_from}x seviyesinden {leverage_down_to}x seviyesine geriledi."
        ),
        "rationale": (
            "Borçluluğun azalması ve kaldıraç oranındaki iyileşme değerleme açısından "
            "olumlu kabul edilir."
        ),
    },
    {
        "label": "positive",
        "difficulty": "clear",
        "category": "shareholder_return",
        "template": (
            "{company}, hisse başına {dividend_tl} TL nakit temettü dağıtma kararı aldı "
            "ve eş zamanlı olarak {buyback_m} milyon TL tutarında pay geri alım programı "
            "başlattığını duyurdu."
        ),
        "rationale": (
            "Temettü ve geri alım birlikte hissedar getirisi açısından güçlü pozitif "
            "sinyal oluşturur."
        ),
    },
    {
        "label": "positive",
        "difficulty": "clear",
        "category": "capacity_expansion",
        "template": (
            "{company}, {city} tesisindeki kapasite artış yatırımı sonrası üretim hacmini "
            "%{capacity_up} yükseltti; yönetim ek kapasitenin ihracat gelirlerini "
            "desteklemesini bekliyor."
        ),
        "rationale": (
            "Kapasite artışı ve ihracat gelirine doğrudan bağlanan etki, yatırımcı için "
            "olumlu finansal beklenti yaratır."
        ),
    },
    {
        "label": "positive",
        "difficulty": "nuanced",
        "category": "margin_improvement",
        "template": (
            "{company}, daha yüksek katma değerli ürün karması sayesinde brüt kâr "
            "marjını %{margin_up_from} seviyesinden %{margin_up_to} seviyesine taşıdı "
            "ve operasyonel kârlılığın yılın kalanında korunacağını bildirdi."
        ),
        "rationale": (
            "Marj artışı ve operasyonel kârlılığın korunacağı mesajı yatırımcı açısından "
            "pozitiftir; ancak cümle yapısı daha nüanslıdır."
        ),
    },
    {
        "label": "negative",
        "difficulty": "clear",
        "category": "earnings_warning",
        "template": (
            "{company}, {period} döneminde net kârının %{profit_drop} gerilediğini açıkladı "
            "ve yıl sonu satış hacmi beklentisini aşağı yönlü güncelledi."
        ),
        "rationale": (
            "Kârlılıktaki düşüş ve aşağı yönlü beklenti revizyonu yatırımcı açısından "
            "açık biçimde negatiftir."
        ),
    },
    {
        "label": "negative",
        "difficulty": "clear",
        "category": "contract_loss",
        "template": (
            "{company}, {country} pazarındaki {contract_m} milyon TL tutarındaki ana "
            "dağıtım sözleşmesinin yenilenmediğini ve sipariş akışında zayıflama "
            "beklediğini duyurdu."
        ),
        "rationale": (
            "Önemli bir ticari sözleşmenin kaybı gelecekteki satışlar için somut negatif "
            "sinyaldir."
        ),
    },
    {
        "label": "negative",
        "difficulty": "clear",
        "category": "liquidity_pressure",
        "template": (
            "{company}, artan işletme sermayesi ihtiyacı nedeniyle {debt_addition_m} milyon TL "
            "ek kısa vadeli kredi kullandı; net borç/FAVÖK oranı {leverage_up_from}x "
            "seviyesinden {leverage_up_to}x seviyesine yükseldi."
        ),
        "rationale": (
            "Likidite baskısı ve kaldıraç artışı yatırımcı açısından negatif finansal "
            "gelişmelerdir."
        ),
    },
    {
        "label": "negative",
        "difficulty": "clear",
        "category": "shutdown_and_fine",
        "template": (
            "{company}, {city} tesisindeki üretimin regülasyon kaynaklı denetim nedeniyle "
            "geçici olarak durdurulduğunu ve {fine_m} milyon TL idari para cezası için "
            "karşılık ayırdığını bildirdi."
        ),
        "rationale": (
            "Üretim kesintisi ve para cezası birlikte şirket değeri için açık negatif "
            "etki taşır."
        ),
    },
    {
        "label": "negative",
        "difficulty": "clear",
        "category": "dividend_cut",
        "template": (
            "{company}, zayıflayan nakit akışı nedeniyle bu yıl temettü dağıtmayacağını "
            "ve planlanan yatırım harcamalarının bir bölümünü ertelediğini duyurdu."
        ),
        "rationale": (
            "Temettü iptali ve harcama ertelemesi finansal zayıflık işareti olduğu için "
            "negatif etiketlenir."
        ),
    },
    {
        "label": "negative",
        "difficulty": "nuanced",
        "category": "margin_pressure",
        "template": (
            "{company}, hammadde ve kur baskısı nedeniyle FAVÖK marjı beklentisini "
            "%{margin_down_from} seviyesinden %{margin_down_to} seviyesine düşürdü; "
            "yönetim ikinci yarıda kârlılıkta baskının sürebileceğini belirtti."
        ),
        "rationale": (
            "Marj beklentisinin aşağı çekilmesi ve devam eden baskı mesajı yatırımcı "
            "açısından negatiftir; ifade görece daha nüanslıdır."
        ),
    },
    {
        "label": "neutral",
        "difficulty": "clear",
        "category": "general_assembly_notice",
        "template": (
            "{company}, olağan genel kurul toplantısının {date} tarihinde yapılacağını "
            "ve gündem maddelerini KAP'ta yayımladı."
        ),
        "rationale": (
            "Rutin genel kurul duyurusu, tek başına yatırımcı açısından doğrudan pozitif "
            "veya negatif değer etkisi taşımaz."
        ),
    },
    {
        "label": "neutral",
        "difficulty": "clear",
        "category": "management_change",
        "template": (
            "{company}, finans direktörlüğüne {person} adlı yöneticinin atandığını duyurdu. "
            "Şirket, değişikliğin mevcut finansal beklentiler üzerinde etkisi bulunmadığını belirtti."
        ),
        "rationale": (
            "Yönetici değişimi tek başına ve finansal etki belirtilmeden verildiği için "
            "nötr kabul edilir."
        ),
    },
    {
        "label": "neutral",
        "difficulty": "clear",
        "category": "routine_filing",
        "template": (
            "{company}, {period} faaliyet raporunu ve bağımsız denetim sürecine ilişkin "
            "takvimi kamuya açıkladı."
        ),
        "rationale": (
            "Faaliyet raporu ve denetim takvimi bildirimi rutin açıklamadır; tek başına "
            "duygusal kutup taşımaz."
        ),
    },
    {
        "label": "neutral",
        "difficulty": "nuanced",
        "category": "investment_review",
        "template": (
            "{company}, {city} bölgesindeki olası yatırım seçeneklerini değerlendirdiğini "
            "ancak proje için henüz bağlayıcı karar alınmadığını bildirdi."
        ),
        "rationale": (
            "Yatırım ihtimali dile getirilse de bağlayıcı karar ve somut finansal etki "
            "olmadığı için nötr etiketlenir."
        ),
    },
    {
        "label": "neutral",
        "difficulty": "nuanced",
        "category": "loan_renewal_unchanged",
        "template": (
            "{company}, mevcut kredi limitini aynı tutar ve benzer koşullarla "
            "{maturity_month} ay uzattı; açıklamada kârlılık beklentilerinde değişiklik "
            "olmadığı belirtildi."
        ),
        "rationale": (
            "Kredi limitinin mevcut koşullarla yenilenmesi önemli yeni pozitif ya da "
            "negatif bilgi içermediği için nötrdür."
        ),
    },
    {
        "label": "neutral",
        "difficulty": "clear",
        "category": "legal_process_update",
        "template": (
            "{company}, devam eden dava dosyasına ilişkin bir sonraki duruşmanın "
            "{date} tarihinde görüleceğini açıkladı; finansal etkisine yönelik yeni bir "
            "gelişme paylaşılmadı."
        ),
        "rationale": (
            "Yeni finansal sonuç içermeyen dava süreci güncellemesi yatırımcı açısından "
            "nötr kabul edilir."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Turkish financial sentiment benchmark."
    )
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=20,
        help="Number of synthetic examples to create per scenario family.",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/finturkbert_benchmark.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--csv-out",
        default="data/finturkbert_benchmark.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def fmt_decimal(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def fmt_tl(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def build_context(family_idx: int, sample_idx: int, attempt: int) -> dict[str, str]:
    rng = random.Random(SEED + family_idx * 10_000 + sample_idx * 97 + attempt)

    margin_up_from = rng.uniform(14.0, 24.0)
    margin_up_to = margin_up_from + rng.uniform(1.4, 5.2)
    margin_down_from = rng.uniform(16.0, 28.0)
    margin_down_to = max(6.0, margin_down_from - rng.uniform(1.8, 6.5))

    leverage_down_from = rng.uniform(1.8, 4.3)
    leverage_down_to = max(0.6, leverage_down_from - rng.uniform(0.5, 1.5))
    leverage_up_from = rng.uniform(1.1, 3.2)
    leverage_up_to = leverage_up_from + rng.uniform(0.5, 1.7)

    return {
        "company": rng.choice(COMPANIES),
        "period": rng.choice(PERIODS),
        "country": rng.choice(COUNTRIES),
        "city": rng.choice(CITIES),
        "person": rng.choice(PEOPLE),
        "date": rng.choice(DATES),
        "profit_growth": str(rng.randint(14, 68)),
        "profit_drop": str(rng.randint(12, 57)),
        "guide_up": str(rng.randint(5, 18)),
        "profit_m": fmt_int(rng.randint(180, 2_800)),
        "contract_m": fmt_int(rng.randint(120, 1_650)),
        "duration_year": str(rng.choice([2, 3, 4, 5, 7])),
        "debt_reduction_m": fmt_int(rng.randint(140, 1_850)),
        "debt_addition_m": fmt_int(rng.randint(140, 1_850)),
        "buyback_m": fmt_int(rng.randint(90, 950)),
        "fine_m": fmt_int(rng.randint(25, 420)),
        "capacity_up": str(rng.randint(10, 62)),
        "dividend_tl": fmt_tl(rng.uniform(0.45, 4.80)),
        "margin_up_from": fmt_decimal(margin_up_from),
        "margin_up_to": fmt_decimal(margin_up_to),
        "margin_down_from": fmt_decimal(margin_down_from),
        "margin_down_to": fmt_decimal(margin_down_to),
        "leverage_down_from": fmt_decimal(leverage_down_from),
        "leverage_down_to": fmt_decimal(leverage_down_to),
        "leverage_up_from": fmt_decimal(leverage_up_from),
        "leverage_up_to": fmt_decimal(leverage_up_to),
        "maturity_month": str(rng.choice([6, 9, 12, 18, 24])),
    }


def generate_rows(samples_per_family: int) -> list[dict[str, str]]:
    rows = []
    seen_texts = set()

    for family_idx, family in enumerate(FAMILIES):
        created = 0
        attempt = 0
        while created < samples_per_family:
            context = build_context(family_idx, created, attempt)
            text = family["template"].format(**context)
            attempt += 1
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append(
                {
                    "text": text,
                    "ground_truth": family["label"],
                    "ground_truth_tr": tr_label(family["label"]),
                    "difficulty": family["difficulty"],
                    "difficulty_tr": tr_difficulty(family["difficulty"]),
                    "category": family["category"],
                    "category_tr": tr_category(family["category"]),
                    "label_rationale": family["rationale"],
                    "source": "synthetic_rule_based",
                }
            )
            created += 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = f"BENCH-{index:04d}"
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "text",
        "ground_truth",
        "ground_truth_tr",
        "difficulty",
        "difficulty_tr",
        "category",
        "category_tr",
        "label_rationale",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    summary = {"ground_truth": {}, "difficulty": {}, "category": {}}
    for row in rows:
        for key in summary:
            value = row[key]
            summary[key][value] = summary[key].get(value, 0) + 1
    return summary


def main() -> None:
    args = parse_args()
    rows = generate_rows(args.samples_per_family)
    jsonl_path = Path(args.jsonl_out)
    csv_path = Path(args.csv_out)
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    summary = summarize(rows)
    print(f"Generated {len(rows)} examples.")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")
    print("Label distribution:", summary["ground_truth"])
    print("Difficulty distribution:", summary["difficulty"])


if __name__ == "__main__":
    main()
