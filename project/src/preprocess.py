import pandas as pd

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)                   # 엑셀 파일 로드
    # print("원본 데이터 로드 완료")
    # print(df.head())   # 앞부분 확인

    # 날짜 변환
    df["date"] = pd.to_datetime(df["date"], errors='coerce')   # 문자열 -> datetime
    # print("날짜 변환 완료")
    # print(df[["date"]].head())

    # user_id 타입 통일
    df["user_id"] = df["user_id"].astype(str) # user_id는 범주형이므로 str 타입 지정
    # print("user_id 타입 변환 완료")
    # print(df[["user_id"]].head())

    # 결제수단 결측/비정상 처리
    df["payment_method"] = df["payment_method"].fillna("Unknown") # 결측 'Unknown'
    df["payment_method"] = df["payment_method"].replace(r"^\s*$", "Unknown", regex=True)
    # print("결제 수단 결측/비정상 처리 완료")
    # print(df["payment_method"].value_counts().head())

    # 할인율 결측/이상치 처리
    df["discount_rate"] = pd.to_numeric(df["discount_rate"], errors='coerce').fillna(0)
    df["discount_rate"] = df["discount_rate"].clip(0, 1) # 0~1 사이로 제한
    # print("할인율 결측/이상치 처리 완료")
    # print(df["discount_rate"].describe())

    # app_time_min 결측/이상치 처리
    df["app_time_min"] = pd.to_numeric(df["app_time_min"], errors='coerce').fillna(0)
    # print("app_time_min 결측/이상치 처리 완료")

    # paid_amount 결측/이상치 처리
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors='coerce').fillna(0)
    # print("paid_amount 결측/이상치 처리 완료")

    # 핵심 식별자 결측 제거
    df = df.dropna(subset=["date", "user_id"])

    # 날짜 순 정렬  
    df = df.sort_values("date").reset_index(drop=True)        
    # print("날짜 순 정렬 완료")

    print("전처리 완료 데이터 shape:", df.shape)
    print(df.head())

    return df

if __name__ == "__main__":
    import os
    # 엑셀 파일 경로 지정 (project 폴더 기준)
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aiml_test_data 1.xlsx")
    df = load_and_clean(data_path)