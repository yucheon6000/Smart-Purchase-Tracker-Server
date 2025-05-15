import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import os

# 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 토크나이저, 라벨 인코더 로드 (최초 1회만)
model = AutoModelForSequenceClassification.from_pretrained(
    "./models/kc_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./models/kc_model")
with open("./models/kc_label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 전처리 함수


def clean_text(text):
    text = re.sub(r"[^\w가-힣\s]", "", text)
    text = re.sub(r"\bP\b|ML|ml|G|g|g\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def categorize_items(purchase_items):
    texts = []
    valid_items = []

    for entry in purchase_items:
        item = entry.get("item", "")
        cleaned = clean_text(item)
        if cleaned and len(cleaned) >= 2:
            texts.append(cleaned)
            valid_items.append(entry)

    if not texts:
        return purchase_items  # 분류할 게 없으면 원본 반환

    inputs = tokenizer(texts, truncation=True, padding=True,
                       max_length=64, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    labels = le.inverse_transform(preds.cpu().numpy())

    for i, entry in enumerate(valid_items):
        entry["category"] = labels[i]

    return purchase_items


categorize_items
