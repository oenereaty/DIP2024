# 사과잎 병해충 구별 모델🍎🍃

## 모델 파일 구성

이 프로젝트에서는 사전 학습된 이미지 분류 모델이 model_directory.zip 형태로 제공됩니다. 이 ZIP 파일에는 다음과 같은 항목이 포함되어 있습니다:

- pytorch_model.bin: 모델 가중치 파일.
- config.json: 모델 구성 파일.
- preprocessor_config.json: 이미지 전처리 설정.
이 파일들을 사용해 특정 이미지 분류 작업을 수행할 수 있습니다.

## 환경 요구사항
모델을 실행하려면 다음 소프트웨어 및 패키지가 필요합니다:
```python
pip install transformers torch pillow requests
```
