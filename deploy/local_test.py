import json
import base64
from lambda_function import lambda_handler


def load_image_as_base64(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def simulate_event(image_base64):
    return {
        "requestContext": {
            "http": {
                "method": "POST"
            }
        },
        "body": json.dumps({
            "image": image_base64
        })
    }


def main():
    # 테스트할 이미지 파일 경로
    image_path = "./test_image.jpg"  # 🔥 여기에 테스트할 이미지 파일 경로 입력

    # 이미지 로드
    image_base64 = load_image_as_base64(image_path)

    # Lambda 이벤트 시뮬레이션
    event = simulate_event(image_base64)
    context = {}  # 빈 context 전달

    # Lambda 핸들러 호출
    response = lambda_handler(event, context)

    print("\n[Lambda 응답 결과]")
    print(json.dumps(json.loads(
        response['body']), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
