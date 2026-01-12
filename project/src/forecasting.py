import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_forecast(daily_sales: pd.DataFrame):
    """
    일별 매출 데이터를 기반으로 Prophet 예측 수행
    """
    # 0. 날짜 타입 변환
    daily_sales = daily_sales.copy()
    daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])

    # 1. Train / Test 분리
    train_df = daily_sales[daily_sales["ds"] < "2024-04-01"]
    test_df  = daily_sales[daily_sales["ds"] >= "2024-04-01"]

    # 2. 모델 학습
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df)

    # 3. 예측 (Train + Test 기간 전체)
    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)

    # 4. 결과 병합 (Test set만 추출)
    test_result = forecast[forecast["ds"] >= "2024-04-01"].copy()
    test_result = test_result[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        test_df[["ds", "y"]],
        on="ds",
        how="inner"
    )
    
    # 결측값 처리 (혹시 모를 경우)
    test_result = test_result.dropna(subset=["y", "yhat"])

    # 5. 평가 지표 계산
    if len(test_result) > 0:
        mae = mean_absolute_error(test_result["y"], test_result["yhat"])
        mape = np.mean(np.abs((test_result["y"] - test_result["yhat"]) / test_result["y"])) * 100
        rmse = np.sqrt(mean_squared_error(test_result["y"], test_result["yhat"]))
    else:
        mae = mape = rmse = np.nan
        print("⚠ 경고: Test set 예측 결과가 없습니다.")
    
    # 전체 결과 (시각화용)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        daily_sales[["ds", "y"]],
        on="ds",
        how="left"
    )

    return model, forecast, result, {"MAE": mae, "MAPE": mape, "RMSE": rmse, "test_result": test_result}

def plot_testset_forecast(result, output_dir: str = "project/outputs", save: bool = True):
    """
    Test set(2024-04-01 ~ 2024-04-30) 구간에서
    실제값 vs 예측값 비교 그래프 및 성능 지표 출력
    """
    import os
    
    # Test set만 추출
    test_result = result[(result["ds"] >= "2024-04-01") & (result["ds"] <= "2024-04-30")].copy()
    test_result = test_result.sort_values("ds")

    # 시각화
    plt.figure(figsize=(14, 6))
    plt.plot(test_result["ds"], test_result["y"], label="Actual", marker="o", linewidth=2, markersize=6)
    plt.plot(test_result["ds"], test_result["yhat"], label="Predicted", marker="x", linewidth=2, markersize=8)
    
    # 신뢰구간 시각화 (있는 경우)
    if "yhat_lower" in test_result.columns and "yhat_upper" in test_result.columns:
        plt.fill_between(test_result["ds"], test_result["yhat_lower"], test_result["yhat_upper"], 
                        alpha=0.2, label="Confidence Interval")
    
    plt.xticks(rotation=45)
    plt.title("Test Set: Actual vs Predicted Sales (2024-04)", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily Sales (KRW)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "daily_sales_forecast.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"매출 예측 시각화 저장 완료: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from preprocess import load_and_clean
    from features import make_daily_sales

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aiml_test_data 1.xlsx")
    df = load_and_clean(data_path)
    daily_sales = make_daily_sales(df)

    model, forecast, result, metrics = run_forecast(daily_sales)
    print(f"\n=== 매출 예측 모델 성능 지표 ===")
    print(f"MAE: {metrics['MAE']:,.0f}")
    print(f"RMSE: {metrics['RMSE']:,.0f}")
    print(f"MAPE: {metrics['MAPE']:.2f}%")

    # Test set 예측 결과 그래프
    plot_testset_forecast(result, save=True)