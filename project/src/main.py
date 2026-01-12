"""
아로아랩스 AI/ML 사전 테스트 - 문제 2: 데이터 분석 및 모델링

main.py 실행 시 다음 과정이 순차적으로 수행됩니다:
1. 데이터 전처리 (Preprocessing)
2. Feature Engineering (파생 변수 생성)
3. 과제 2-1: 고객 세분화 (Clustering)
4. 과제 2-2: 매출 예측 (Time-Series Forecasting)
5. 평가 및 시각화
"""

import os
import sys
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from preprocess import load_and_clean
from features import make_user_features, make_daily_sales
from clustering import run_clustering, visualize_clusters_tsne, print_cluster_summary
from forecasting import run_forecast, plot_testset_forecast


def main():
    """메인 실행 함수"""
    print("="*80)
    print("AI/ML 사전 테스트 - 문제 2: 데이터 분석 및 모델링")
    print("="*80)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ============================================
    # 1. 데이터 전처리 (Preprocessing)
    # ============================================
    print("[1단계] 데이터 전처리 시작...")
    data_path = os.path.join(project_root, "aiml_test_data 1.xlsx")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    df = load_and_clean(data_path)
    print(f"✓ 전처리 완료: {df.shape[0]:,}행, {df.shape[1]}열\n")
    
    # ============================================
    # 2. Feature Engineering (파생 변수 생성)
    # ============================================
    print("[2단계] Feature Engineering 시작...")
    
    # 2-1. 고객 세분화용 사용자 파생 변수
    user_features = make_user_features(df)
    print(f"✓ 사용자 파생 변수 생성 완료: {len(user_features):,}명의 고객")
    
    # 2-2. 시계열 예측용 일별 매출 데이터
    daily_sales = make_daily_sales(df)
    print(f"✓ 일별 매출 데이터 생성 완료: {len(daily_sales):,}일\n")
    
    # ============================================
    # 3. 과제 2-1: 고객 세분화 (Clustering)
    # ============================================
    print("[3단계] 과제 2-1: 고객 세분화 (Clustering) 시작...")
    
    # 3-1. 군집화 수행 (3개 이상의 군집)
    n_clusters = 4
    clustered_df = run_clustering(user_features, n_clusters=n_clusters)
    print(f"✓ K-Means 군집화 완료: {n_clusters}개 군집")
    print(f"  군집별 고객 수:")
    for cluster_id in sorted(clustered_df['cluster'].unique()):
        count = len(clustered_df[clustered_df['cluster'] == cluster_id])
        pct = count / len(clustered_df) * 100
        print(f"    - 군집 {cluster_id}: {count:,}명 ({pct:.1f}%)")
    
    # 3-2. 군집별 통계 요약 출력
    cluster_stats = print_cluster_summary(clustered_df)
    
    # 3-3. t-SNE 시각화
    output_dir = os.path.join(project_root, "outputs")
    visualize_clusters_tsne(clustered_df, output_dir=output_dir)
    
    print(f"✓ 고객 세분화 분석 완료\n")
    
    # ============================================
    # 4. 과제 2-2: 매출 예측 (Time-Series Forecasting)
    # ============================================
    print("[4단계] 과제 2-2: 매출 예측 (Time-Series Forecasting) 시작...")
    
    # 4-1. 모델 학습 및 예측
    # Train: 2023-01-01 ~ 2024-03-31
    # Test: 2024-04-01 ~ 2024-04-30
    model, forecast, result, metrics = run_forecast(daily_sales)
    
    # 4-2. 성능 지표 출력
    print(f"\n✓ 매출 예측 모델 학습 및 평가 완료")
    print(f"\n  [성능 지표]")
    print(f"    - MAE (Mean Absolute Error): {metrics['MAE']:,.0f}원")
    print(f"    - RMSE (Root Mean Squared Error): {metrics['RMSE']:,.0f}원")
    print(f"    - MAPE (Mean Absolute Percentage Error): {metrics['MAPE']:.2f}%")
    
    # 4-3. 시각화 및 저장
    plot_testset_forecast(result, output_dir=output_dir, save=True)
    
    print(f"✓ 매출 예측 분석 완료\n")
    
    # ============================================
    # 5. 최종 요약
    # ============================================
    print("="*80)
    print("전체 분석 프로세스 완료!")
    print("="*80)
    print(f"\n생성된 결과물:")
    print(f"  - 고객 세분화 시각화: {os.path.join(output_dir, 'customer_segmentation_tsne.png')}")
    print(f"  - 매출 예측 시각화: {os.path.join(output_dir, 'daily_sales_forecast.png')}")
    print(f"\n실행 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)