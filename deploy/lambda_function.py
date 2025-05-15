import base64
import json
from ocr_utils import *
from categorize_items import categorize_items


def lambda_handler(event, context):
    method = event['requestContext']['http']['method']
    if method == 'OPTIONS':
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": ""
        }

    try:
        body = json.loads(event['body'])
        image_data = body.get('image')

        if image_data and image_data.startswith('data:image'):

            header, b64data = image_data.split(',', 1)
            img_bytes = base64.b64decode(b64data)

            # OCR
            texts = detect_text_from_bytes(img_bytes)
            full_text = texts[0].description if texts else ""

            if not full_text:
                raise Exception("OCR 결과 없음")

            # GPT
            gpt_response = ask_gpt_extract_dicts_from_full_text(full_text)
            purchase_items = json.loads(gpt_response)

            # 카테고리 분류 추가
            purchase_items = categorize_items(purchase_items)

            result_msg = "구매 항목 + 카테고리 추출 완료"

        else:
            purchase_items = []
            result_msg = "이미지 데이터 없음 또는 잘못된 형식"
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }

    return {
        "statusCode": 200,
        "headers": {"Access-Control-Allow-Origin": "*"},
        "body": json.dumps({"result": result_msg, "purchaseItems": purchase_items})
    }
