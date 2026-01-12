import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os

def run_clustering(user_features: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """사용자 파생변수를 기반으로 KMeans 군집 수행"""

    # 1. 클러스터링에 사용할 컬럼 선택
    feature_cols = [
        "purchase_cnt",
        "total_paid",
        "avg_paid",
        "avg_discount",
        "discount_usage_rate",
        "avg_app_time",
        "recency_days"
    ]

    X = user_features[feature_cols]

    # 2. 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. KMeans 군집
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(X_scaled)

    # 4. 결과 병합
    result = user_features.copy()
    result["cluster"] = clusters

    return result

def compare_k_distribution(user_features, k_list=[3,4,5,6]):
    """군집 개수(k)별 분포 비교"""
    feature_cols = [
        "purchase_cnt", "total_paid", "avg_paid",
        "avg_discount", "discount_usage_rate",
        "avg_app_time", "recency_days"
    ]

    X = user_features[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        print(f"\n=== K = {k} ===")
        print(pd.Series(labels).value_counts().sort_index())


def visualize_clusters_tsne(clustered_df: pd.DataFrame, output_dir: str = "project/outputs", n_components: int = 2, random_state: int = 42):
    """t-SNE를 활용한 군집 결과 시각화"""
    feature_cols = [
        "purchase_cnt", "total_paid", "avg_paid",
        "avg_discount", "discount_usage_rate",
        "avg_app_time", "recency_days"
    ]
    
    X = clustered_df[feature_cols]
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE 차원 축소
    print("t-SNE 차원 축소 진행 중...")
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # 시각화
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clustered_df['cluster'], cmap='viridis', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Customer Segmentation Visualization (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.tight_layout()
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "customer_segmentation_tsne.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE 시각화 저장 완료: {output_path}")
    plt.close()
    
    return X_tsne


def print_cluster_summary(clustered_df: pd.DataFrame):
    """군집별 통계 요약 출력 (Mean)"""
    feature_cols = [
        "purchase_cnt", "total_paid", "avg_paid",
        "avg_discount", "discount_usage_rate",
        "avg_app_time", "recency_days"
    ]
    
    cluster_stats = clustered_df.groupby('cluster')[feature_cols].agg(['mean']).round(1)
    
    print("\n[Cluster Summary - Mean]")
    print(cluster_stats)
    
    return cluster_stats

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from preprocess import load_and_clean
    from features import make_user_features

    # 1. 데이터 로드
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "aiml_test_data 1.xlsx")
    df = load_and_clean(data_path)

    # 2. 사용자 파생변수 생성
    user_features = make_user_features(df)

    # 1️⃣ k 분포 비교
    compare_k_distribution(user_features, k_list=[3,4,5,6])

    # 2️⃣ 선택한 k로 최종 군집
    clustered_df = run_clustering(user_features, n_clusters=4)

    # 3. 결과 확인
    print(clustered_df[["user_id", "cluster"]].head())
    print(clustered_df["cluster"].value_counts())
    
    # 4. 군집별 통계 요약 출력
    cluster_stats = print_cluster_summary(clustered_df)
    
    # 5. t-SNE 시각화
    visualize_clusters_tsne(clustered_df)