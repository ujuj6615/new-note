import pandas as pd

def make_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """고객 세분화를 위한 사용자 단위 파생 변수 생성"""

    # 1. 사용자 단위 집계
    user_features = (
        df.groupby("user_id")
          .agg(
              purchase_cnt=("paid_amount", "count"),          # 구매 횟수
              total_paid=("paid_amount", "sum"),              # 총 결제금액
              avg_paid=("paid_amount", "mean"),               # 평균 결제금액
              avg_discount=("discount_rate", "mean"),         # 평균 할인율
              discount_usage_rate=("discount_rate", lambda x: (x > 0).mean()),  # 할인 사용 비율
              avg_app_time=("app_time_min", "mean"),          # 평균 앱 사용 시간
              last_purchase_date=("date", "max")              # 최근 구매 경과일(일)
          )
          .reset_index()
    )

    # 2. Recency 계산 (기준일자 = 데이터 내 최대 날짜)
    ref_date = df["date"].max()
    user_features["recency_days"] = (
        ref_date - user_features["last_purchase_date"]
    ).dt.days

    return user_features

def make_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """일별 총 매출 데이터 생성 (시계열 예측용)"""

    daily_sales = (
        df.groupby(df["date"].dt.date) # 날짜 단위로 그룹화
          .agg(daily_total_paid=("paid_amount", "sum")) # 일별 총 결제 금액 합계
          .reset_index() # 그룹화 결과를 DataFrame으로 다시 정리
          .rename(columns={"date": "ds", "daily_total_paid": "y"}) # 시계열 예측 형식 맞추기
    )

    return daily_sales

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from preprocess import load_and_clean

    # 1. 데이터 로드 & 전처리
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aiml_test_data 1.xlsx")
    df = load_and_clean(data_path)

    # 2. 사용자 파생 변수 생성
    user_features = make_user_features(df)
    print("User features sample:")
    print(user_features.head())
    print(user_features.describe())

    # 3. 일별 매출 데이터 생성
    daily_sales = make_daily_sales(df)
    print("Daily sales sample:")
    print(daily_sales.head())