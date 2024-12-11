# 사과잎 병해충 구별 모델🍎🍃



## 1. 모델 파일 구성

이 프로젝트에서는 사전 학습된 이미지 분류 모델이 `model_directory.zip` 형태로 제공됩니다.
이 ZIP 파일에는 다음과 같은 항목이 포함되어 있습니다:

- `pytorch_model.bin`: 모델 가중치 파일.
- `config.json`: 모델 구성 파일.
- `preprocessor_config.json`: 이미지 전처리 설정.
이 파일들을 사용해 특정 이미지 분류 작업을 수행할 수 있습니다.

## 2. 환경 요구사항
모델을 실행하려면 다음 소프트웨어 및 패키지가 필요합니다:

- Python 3.8 이상
- 주요 라이브러리:
```python
pip install transformers torch pillow requests
```

## 3. Jupyter Notebook에서 모델 사용하기
Jupyter Notebook을 실행합니다.
제공된 `model_directory.zip` 파일을 **Notebook 세션 저장소**에 업로드합니다.
업로드된 ZIP 파일을 다음 코드를 사용해 압축 해제하고 모델을 로드합니다:
```python
import zipfile
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# ZIP 파일 압축 해제
zip_path = "model_directory.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./")  # 현재 디렉토리에 압축 해제

# 모델 로드
model_directory = "model_directory"
model = AutoModelForImageClassification.from_pretrained(model_directory)
image_processor = AutoFeatureExtractor.from_pretrained(model_directory)
```

다음 코드를 사용해 외부 이미지 URL을 처리하고 예측 결과를 확인합니다:
```python
from PIL import Image
import torch
import requests
from io import BytesIO

# 외부 이미지 URL 처리 함수
def process_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception("이미지를 가져올 수 없습니다.")
    
    image = Image.open(BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    
    print(f"예측 결과: 라벨 {predictions}")

# 이미지 넣기
image_url = "https://example.com/sample_image.jpg"
process_image_from_url(image_url)
```


## 4. 라벨 정보 및 해결책
모델이 예측한 라벨에 대한 정보와 해결책은 아래와 같습니다:

| 라벨 | 클래스 이름                 | 해결책                                                                                 |
|------|-----------------------------|---------------------------------------------------------------------------------------|
| 0    | `Apple___Apple_scab`       | 감염된 잎을 제거하고 살균제를 사용해 추가 확산을 방지하세요.                          |
| 1    | `Apple___Black_rot`        | 감염된 가지와 과일을 제거하고 구리 기반 살균제를 사용해 예방하세요.                   |
| 2    | `Apple___Cedar_apple_rust` | 근처의 주니퍼 나무를 제거하거나 방제제를 사용해 증상을 완화할 수 있습니다.            |
| 3    | `Apple___healthy`          | 사과 나무는 건강합니다. 추가 조치가 필요하지 않습니다!                               |


## 5. openCV 이용

