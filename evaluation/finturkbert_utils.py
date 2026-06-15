from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_ID = "ff112/FinTurkBERT"
LABELS = ["negative", "neutral", "positive"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
LABEL_TR = {
    "negative": "negatif",
    "neutral": "nötr",
    "positive": "pozitif",
}
DIFFICULTY_TR = {
    "clear": "açık",
    "nuanced": "nüanslı",
}
CATEGORY_TR = {
    "earnings_upgrade": "kâr artışı ve beklenti yükseltimi",
    "contract_win": "yeni sözleşme kazanımı",
    "deleveraging": "borçluluk azalması",
    "shareholder_return": "temettü ve geri alım",
    "capacity_expansion": "kapasite artışı",
    "margin_improvement": "marj iyileşmesi",
    "earnings_warning": "kâr düşüşü ve uyarı",
    "contract_loss": "sözleşme kaybı",
    "liquidity_pressure": "likidite baskısı",
    "shutdown_and_fine": "üretim duruşu ve ceza",
    "dividend_cut": "temettü iptali",
    "margin_pressure": "marj baskısı",
    "general_assembly_notice": "genel kurul duyurusu",
    "management_change": "yönetim değişikliği",
    "routine_filing": "rutin bildirim",
    "investment_review": "yatırım değerlendirmesi",
    "loan_renewal_unchanged": "kredi yenileme değişikliksiz",
    "legal_process_update": "hukuki süreç güncellemesi",
}


def tr_label(label: str) -> str:
    return LABEL_TR.get(label, label)


def tr_difficulty(level: str) -> str:
    return DIFFICULTY_TR.get(level, level)


def tr_category(category: str) -> str:
    return CATEGORY_TR.get(category, category)


def load_model(cache_dir: str):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
    )
    model.eval()
    return tokenizer, model


def predict_texts(
    texts: list[str],
    tokenizer,
    model,
    max_length: int = 64,
    batch_size: int = 16,
):
    rows = []
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1).tolist()

        for text, pred_id, scores in zip(
            batch_texts,
            predicted_ids,
            probabilities.tolist(),
        ):
            rows.append(
                {
                    "text": text,
                    "label": ID2LABEL[pred_id],
                    "label_tr": tr_label(ID2LABEL[pred_id]),
                    "scores": {
                        ID2LABEL[idx]: round(score, 6)
                        for idx, score in enumerate(scores)
                    },
                }
            )
    return rows
