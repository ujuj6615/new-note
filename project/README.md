# 프로젝트 설명

이 프로젝트는 이커머스 서비스의 로그 데이터를 활용하여 고객 세분화(Clustering)와 일별 매출 예측(Time-Series Forecasting)을 수행하는 AI/ML 분석 파이프라인입니다.

## 폴더 구조

- src/ : 데이터 전처리, 피처 엔지니어링, 클러스터링, 예측 등 주요 파이프라인 코드
- requirements.txt : 실행에 필요한 Python 패키지 목록
- (데이터 파일, 결과물 등은 별도)

## 실행 방법

1. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

2. 메인 파이프라인 실행

```bash
python src/main.py
```

- main.py 실행 시 전처리 → 피처 엔지니어링 → 군집화 → 예측 → 평가가 순차적으로 자동 수행됩니다.
- 주요 결과물(군집 요약, t-SNE 시각화, 예측 결과 등)은 outputs/ 폴더에 저장됩니다.

## 주요 기능

- **고객 세분화**: K-Means 기반, t-SNE 시각화, 군집별 특성 분석
- **매출 예측**: Prophet 기반 시계열 예측, Test set(2024-04) 기준 성능 평가(MAE, RMSE, MAPE)
- **비즈니스 인사이트**: 군집별/예측 기반 전략 및 실행 방안 제안

## 환경
- Python 3.8 이상 권장
- requirements.txt 참고

## 문의
- 과제 관련 문의는 README 상단 안내 또는 제출처로 연락 바랍니다.
