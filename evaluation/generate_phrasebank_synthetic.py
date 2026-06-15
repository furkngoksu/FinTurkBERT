#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path


SEED = 20260329
COMPANIES = [
    "Aksa Enerji",
    "Anatolia Gıda",
    "Bora Lojistik",
    "Delta Yatırım Holding",
    "Eksen Teknoloji",
    "Formel Kimya",
    "Güney Perakende",
    "Hedef Savunma",
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
    "Mısır",
    "Polonya",
    "Romanya",
]
CITIES = [
    "Ankara",
    "Bursa",
    "Gaziantep",
    "İzmir",
    "Kocaeli",
    "Manisa",
]
DATES = [
    "7 Nisan 2026",
    "18 Nisan 2026",
    "5 Mayıs 2026",
    "14 Mayıs 2026",
    "28 Mayıs 2026",
]
PEOPLE = [
    "Ayşe Demir",
    "Cem Arslan",
    "Elif Korkmaz",
    "Mert Sevinç",
    "Nazlı Acar",
    "Selin Aksoy",
]

LABEL_TR = {
    "negative": "negatif",
    "neutral": "nötr",
    "positive": "pozitif",
}

FAMILIES = [
    {
        "label": "positive",
        "category_id": "clear_profit_growth",
        "category": "açık kâr artışı",
        "difficulty": "clear",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, {period} döneminde net kârını %{profit_growth} artırarak "
            "{profit_m} milyon TL'ye çıkardı ve yıl sonu gelir beklentisini %{guide_up} yukarı revize etti."
        ),
        "annotation_note": (
            "Yatırımcı açısından net kâr artışı ve yukarı yönlü beklenti revizyonu açık pozitif değerdir."
        ),
    },
    {
        "label": "positive",
        "category_id": "clear_contract_gain",
        "category": "açık sözleşme kazanımı",
        "difficulty": "clear",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, {country} pazarında {contract_m} milyon TL tutarında "
            "{duration_year} yıllık satış sözleşmesi imzaladı; anlaşmanın gelecek 12 ay gelirlerine "
            "doğrudan katkı sağlaması bekleniyor."
        ),
        "annotation_note": (
            "Somut büyüklüğü olan yeni sözleşme yatırımcı açısından açık pozitif kabul edilir."
        ),
    },
    {
        "label": "positive",
        "category_id": "clear_deleveraging",
        "category": "açık borçluluk azalması",
        "difficulty": "clear",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, varlık satışı sonrasında net borcunu {debt_m} milyon TL azalttı ve "
            "net borç/FAVÖK oranının {ratio_from}x seviyesinden {ratio_to}x seviyesine gerilediğini açıkladı."
        ),
        "annotation_note": (
            "Borçluluğun düşmesi ve kaldıraç oranındaki iyileşme yatırımcı açısından pozitiftir."
        ),
    },
    {
        "label": "positive",
        "category_id": "clear_margin_improvement",
        "category": "açık marj iyileşmesi",
        "difficulty": "clear",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, ürün karmasındaki iyileşme sayesinde brüt kâr marjını "
            "%{margin_from} seviyesinden %{margin_to} seviyesine yükseltti ve operasyonel "
            "kârlılığın yılın kalanında korunacağını bildirdi."
        ),
        "annotation_note": (
            "Açık marj iyileşmesi ve korunacak kârlılık beklentisi yatırımcı açısından pozitiftir."
        ),
    },
    {
        "label": "positive",
        "category_id": "clear_shareholder_return",
        "category": "açık hissedar getirisi",
        "difficulty": "clear",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, hisse başına {dividend_tl} TL nakit temettü dağıtacağını ve "
            "{buyback_m} milyon TL tutarında pay geri alım programı başlatacağını açıkladı."
        ),
        "annotation_note": (
            "Temettü ve geri alım birlikte yatırımcı açısından açık pozitif değer sinyali üretir."
        ),
    },
    {
        "label": "positive",
        "category_id": "clear_capacity_expansion",
        "category": "açık kapasite artışı",
        "difficulty": "clear",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, {city} tesisindeki yatırım sonrası üretim kapasitesini %{capacity_up} artırdı "
            "ve ek kapasitenin ihracat gelirlerine katkı vermesini beklediğini duyurdu."
        ),
        "annotation_note": (
            "Kapasite artışı ve buna bağlanan gelir beklentisi yatırımcı açısından pozitiftir."
        ),
    },
    {
        "label": "positive",
        "category_id": "clear_rating_upgrade",
        "category": "açık kredi notu iyileşmesi",
        "difficulty": "nuanced",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, {period} sonuçlarının ardından uzun vadeli kredi notunun yükseltildiğini "
            "ve finansman maliyetlerinde iyileşme beklediğini açıkladı."
        ),
        "annotation_note": (
            "Kredi notu artışı ve daha düşük finansman maliyeti beklentisi yatırımcı açısından pozitiftir."
        ),
    },
    {
        "label": "neutral",
        "category_id": "procedural_disclosure",
        "category": "prosedürel bildirim",
        "difficulty": "clear",
        "surface_tone": "procedural",
        "hard_case": False,
        "template": (
            "{company}, olağan genel kurul toplantısının {date} tarihinde yapılacağını "
            "ve gündem maddelerini kamuya açıkladı."
        ),
        "annotation_note": (
            "Prosedürel bildirim yatırımcı açısından doğrudan value-relevant pozitif ya da negatif etki taşımaz."
        ),
    },
    {
        "label": "neutral",
        "category_id": "vague_optimism",
        "category": "belirsiz kurumsal iyimserlik",
        "difficulty": "nuanced",
        "surface_tone": "positive_like",
        "hard_case": False,
        "template": (
            "{company}, {period} sonrasında dijital dönüşüm programının orta vadede verimliliği "
            "desteklemesini beklediğini ancak mevcut finansal beklentilerde değişiklik olmadığını belirtti."
        ),
        "annotation_note": (
            "Belirsiz ve dolaylı kurumsal iyimserlik, açık finansal etki yoksa neutral etiketlenir."
        ),
    },
    {
        "label": "neutral",
        "category_id": "non_binding_review",
        "category": "bağlayıcı olmayan değerlendirme",
        "difficulty": "nuanced",
        "surface_tone": "neutral_like",
        "hard_case": False,
        "template": (
            "{company}, {city} bölgesindeki yatırım seçeneklerini değerlendirdiğini ancak proje için "
            "henüz bağlayıcı karar alınmadığını bildirdi."
        ),
        "annotation_note": (
            "Bağlayıcı karar ve net finansal etki olmadığı için yatırımcı açısından neutral kabul edilir."
        ),
    },
    {
        "label": "neutral",
        "category_id": "management_change_no_impact",
        "category": "finansal etkisiz yönetim değişimi",
        "difficulty": "clear",
        "surface_tone": "neutral_like",
        "hard_case": False,
        "template": (
            "{company}, finans direktörlüğüne {person} adlı yöneticinin atandığını duyurdu "
            "ve mevcut finansal beklentilerde herhangi bir değişiklik olmadığını belirtti."
        ),
        "annotation_note": (
            "Yönetim değişimi tek başına ve finansal beklentiler değişmiyorsa neutral kabul edilir."
        ),
    },
    {
        "label": "neutral",
        "category_id": "unchanged_financing_rollover",
        "category": "koşulları değişmeyen finansman yenilemesi",
        "difficulty": "nuanced",
        "surface_tone": "neutral_like",
        "hard_case": False,
        "template": (
            "{company}, mevcut kredi limitini {maturity_month} ay benzer vade ve maliyet koşullarıyla "
            "yenilediğini ve işlem nedeniyle kârlılık beklentilerinde değişiklik olmadığını açıkladı."
        ),
        "annotation_note": (
            "Yeni değer yaratan veya bozan bir unsur içermeyen finansman yenilemesi neutral etiketlenir."
        ),
    },
    {
        "label": "neutral",
        "category_id": "legal_process_update",
        "category": "finansal etkisiz hukuki süreç güncellemesi",
        "difficulty": "clear",
        "surface_tone": "procedural",
        "hard_case": False,
        "template": (
            "{company}, devam eden dava dosyasına ilişkin bir sonraki duruşmanın {date} tarihinde "
            "görüleceğini duyurdu ve finansal etkiye dair yeni bir gelişme paylaşmadı."
        ),
        "annotation_note": (
            "Finansal sonuç içermeyen dava süreci güncellemesi neutral kabul edilir."
        ),
    },
    {
        "label": "neutral",
        "category_id": "routine_report_release",
        "category": "rutin rapor yayımlama",
        "difficulty": "clear",
        "surface_tone": "procedural",
        "hard_case": False,
        "template": (
            "{company}, {period} faaliyet raporunu ve yatırımcı sunumunu kamuya açıkladı; "
            "mevcut beklentilerde değişiklik paylaşmadı."
        ),
        "annotation_note": (
            "Yeni value-relevant bilgi içermeyen rutin rapor yayımlama işlemi neutral kabul edilir."
        ),
    },
    {
        "label": "negative",
        "category_id": "clear_earnings_warning",
        "category": "açık kâr düşüşü uyarısı",
        "difficulty": "clear",
        "surface_tone": "negative_like",
        "hard_case": False,
        "template": (
            "{company}, {period} döneminde net kârının %{profit_drop} gerilediğini açıkladı "
            "ve yıl sonu satış hacmi beklentisini aşağı yönlü güncelledi."
        ),
        "annotation_note": (
            "Kâr düşüşü ve aşağı yönlü beklenti revizyonu yatırımcı açısından açık negatiftir."
        ),
    },
    {
        "label": "negative",
        "category_id": "clear_production_halt",
        "category": "açık üretim duruşu ve ceza",
        "difficulty": "clear",
        "surface_tone": "negative_like",
        "hard_case": False,
        "template": (
            "{company}, {city} tesisindeki üretimin denetim süreci nedeniyle geçici olarak "
            "durdurulduğunu ve {fine_m} milyon TL idari para cezası için karşılık ayırdığını açıkladı."
        ),
        "annotation_note": (
            "Üretim kesintisi ve para cezası birlikte yatırımcı açısından açık negatif sinyaldir."
        ),
    },
    {
        "label": "negative",
        "category_id": "hard_liquidity_pressure",
        "category": "olumlu dille likidite baskısı",
        "difficulty": "hard",
        "surface_tone": "positive_like",
        "hard_case": True,
        "template": (
            "{company}, bilanço esnekliğini korumak ve nakit yönetimini desteklemek amacıyla "
            "{debt_m} milyon TL ek kısa vadeli kredi kullandı; net borç/FAVÖK oranı "
            "{ratio_up_from}x seviyesinden {ratio_up_to}x seviyesine yükseldi."
        ),
        "annotation_note": (
            "Dil yapısı olumlu görünse de artan kısa vadeli borç ve kötüleşen kaldıraç oranı yatırımcı açısından negatiftir."
        ),
    },
    {
        "label": "negative",
        "category_id": "hard_dividend_cut",
        "category": "olumlu dille temettü iptali",
        "difficulty": "hard",
        "surface_tone": "positive_like",
        "hard_case": True,
        "template": (
            "{company}, {period} sonrasında bilanço dayanıklılığını korumak amacıyla bu yıl temettü "
            "dağıtmayacağını ve yatırım harcamalarının bir bölümünü sonraki döneme kaydıracağını duyurdu."
        ),
        "annotation_note": (
            "Koruyucu dil kullanılmasına rağmen temettü iptali ve harcama ertelemesi yatırımcı açısından negatiftir."
        ),
    },
    {
        "label": "negative",
        "category_id": "hard_contract_loss",
        "category": "olumlu dille sözleşme kaybı",
        "difficulty": "hard",
        "surface_tone": "positive_like",
        "hard_case": True,
        "template": (
            "{company}, portföy verimliliğini artırma süreci kapsamında {country} pazarındaki "
            "{contract_m} milyon TL tutarındaki ana dağıtım sözleşmesinin yenilenmediğini ve "
            "sipariş akışında zayıflama beklediğini açıkladı."
        ),
        "annotation_note": (
            "Yeniden dengeleme dili olumlu görünse de önemli sözleşme kaybı yatırımcı açısından negatiftir."
        ),
    },
    {
        "label": "negative",
        "category_id": "hard_rights_issue",
        "category": "olumlu dille bedelli sermaye artırımı",
        "difficulty": "hard",
        "surface_tone": "positive_like",
        "hard_case": True,
        "template": (
            "{company}, sermaye yapısını güçlendirmek ve büyüme esnekliğini artırmak amacıyla "
            "%{dilution_pct} oranında bedelli sermaye artırımı planladığını duyurdu."
        ),
        "annotation_note": (
            "Dil olumlu olsa da hissedar seyrelmesi yaratan bedelli sermaye artırımı yatırımcı açısından negatiftir."
        ),
    },
    {
        "label": "negative",
        "category_id": "hard_inventory_writeoff",
        "category": "olumlu dille değer düşüklüğü",
        "difficulty": "hard",
        "surface_tone": "positive_like",
        "hard_case": True,
        "template": (
            "{company}, portföy sadeleştirme programı kapsamında {writeoff_m} milyon TL stok ve varlık "
            "değer düşüklüğü gideri kaydettiğini açıkladı."
        ),
        "annotation_note": (
            "Portföy sadeleştirme dili olumlu görünse de değer düşüklüğü gideri yatırımcı açısından negatiftir."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PhraseBank-style synthetic Turkish financial sentiment data."
    )
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=40,
        help="Number of examples to generate per scenario family.",
    )
    parser.add_argument(
        "--jsonl-out",
        default="data/phrasebank_viewpoint_synthetic.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--csv-out",
        default="data/phrasebank_viewpoint_synthetic.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def fmt_decimal(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def build_context(family_idx: int, sample_idx: int, attempt: int) -> dict[str, str]:
    rng = random.Random(SEED + family_idx * 10_000 + sample_idx * 97 + attempt)

    ratio_base = rng.uniform(1.2, 2.8)
    ratio_up = ratio_base + rng.uniform(0.4, 1.6)
    ratio_down = max(0.6, ratio_base - rng.uniform(0.4, 1.3))

    return {
        "company": rng.choice(COMPANIES),
        "period": rng.choice(PERIODS),
        "country": rng.choice(COUNTRIES),
        "city": rng.choice(CITIES),
        "date": rng.choice(DATES),
        "person": rng.choice(PEOPLE),
        "profit_growth": str(rng.randint(16, 62)),
        "guide_up": str(rng.randint(5, 18)),
        "profit_m": fmt_int(rng.randint(180, 2_300)),
        "contract_m": fmt_int(rng.randint(150, 1_500)),
        "duration_year": str(rng.choice([2, 3, 4, 5])),
        "maturity_month": str(rng.choice([6, 9, 12, 18, 24])),
        "debt_m": fmt_int(rng.randint(180, 1_600)),
        "buyback_m": fmt_int(rng.randint(120, 900)),
        "fine_m": fmt_int(rng.randint(25, 360)),
        "writeoff_m": fmt_int(rng.randint(40, 420)),
        "capacity_up": str(rng.randint(10, 55)),
        "profit_drop": str(rng.randint(12, 54)),
        "dilution_pct": str(rng.randint(20, 180)),
        "dividend_tl": f"{rng.uniform(0.45, 4.80):.2f}".replace(".", ","),
        "margin_from": fmt_decimal(rng.uniform(14.0, 24.0)),
        "margin_to": fmt_decimal(rng.uniform(24.5, 33.0)),
        "ratio_from": fmt_decimal(ratio_base),
        "ratio_to": fmt_decimal(ratio_down),
        "ratio_up_from": fmt_decimal(ratio_base),
        "ratio_up_to": fmt_decimal(ratio_up),
    }


def generate_rows(samples_per_family: int) -> list[dict[str, object]]:
    rows = []
    seen_texts = set()

    for family_idx, family in enumerate(FAMILIES):
        created = 0
        attempt = 0
        while created < samples_per_family:
            if attempt > samples_per_family * 800:
                raise RuntimeError(
                    f"Could not generate enough unique rows for {family['category_id']}. "
                    "Increase template variety or lower samples-per-family."
                )
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
                    "ground_truth_tr": LABEL_TR[family["label"]],
                    "category_id": family["category_id"],
                    "category": family["category"],
                    "difficulty": family["difficulty"],
                    "surface_tone": family["surface_tone"],
                    "hard_case": family["hard_case"],
                    "viewpoint": "investor_impact",
                    "annotation_note": family["annotation_note"],
                }
            )
            created += 1

    rng = random.Random(SEED)
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = f"PHRASE-{index:04d}"
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "text",
        "ground_truth",
        "ground_truth_tr",
        "category_id",
        "category",
        "difficulty",
        "surface_tone",
        "hard_case",
        "viewpoint",
        "annotation_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    summary = {
        "ground_truth": {},
        "category": {},
        "surface_tone": {},
        "hard_case": {},
    }
    for row in rows:
        for key in summary:
            value = str(row[key])
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
    print("Category distribution:", summary["category"])
    print("Surface tone distribution:", summary["surface_tone"])
    print("Hard case distribution:", summary["hard_case"])


if __name__ == "__main__":
    main()
