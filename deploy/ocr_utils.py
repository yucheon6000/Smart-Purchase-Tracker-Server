import os
import json
from openai import OpenAI
from google.cloud import vision
from google.oauth2 import service_account

# --- [1] OpenAI 클라이언트 생성 ---
openai_api_key = ""
openai_client = OpenAI(api_key=openai_api_key)

# --- [2] Google Vision 클라이언트 생성 (직접 키 지정) ---
# GCP 서비스 계정 키 파일 경로
# 🔥 여기 수정: 서비스 계정 키 JSON 파일 경로
gcp_key_path = ".json"

credentials = service_account.Credentials.from_service_account_file(
    gcp_key_path)
gcv_client = vision.ImageAnnotatorClient(credentials=credentials)

# --- [3] OCR 함수 ---


def detect_text_from_bytes(image_bytes):
    image = vision.Image(content=image_bytes)
    response = gcv_client.text_detection(image=image)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(
            f"{response.error.message}\nMore info: https://cloud.google.com/apis/design/errors")
    return texts

# --- [4] GPT 추출 함수 ---


def ask_gpt_extract_dicts_from_full_text(full_text):
    prompt = f"""다음은 OCR로 추출한 영수증 전체 텍스트야.
여기서 결제 항목만 찾아서 다음과 같은 dict 형태 리스트로 정리해줘:
[
  {{"item": "상품명", "count": 개수(int), "price_per_one": 개당 가격(int), "price": 가격(int), "category": ""}},
  ...
]

답변을 바로 변환해서 사용할 것이기에 답변은 오직 json형태로만 줘야 함 (사족 X)

OCR 텍스트:
\"\"\"
{full_text}
\"\"\"
"""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
