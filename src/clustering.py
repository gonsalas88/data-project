import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_clustering(input_path='data/cleaned_data.csv', output_dir='clustering_results'):
    """
    Groups patients into 4 clusters and summarizes the average result of each group.
    """
    print("\n--- TASK 4: PATTERN IDENTIFICATION (CLUSTERING) ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    # Features related to patient profile and diet behavior.
    features = ['age', 'baseline_bmi', 'motivation_score', 'mean_adherence_pct']
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running K-Means algorithm (k=4)...")
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df['cluster'])
    print(f"Silhouette score: {score:.4f}")

    summary = df.groupby('cluster')[features + ['weight_change_kg_6m']].mean()
    summary['patient_count'] = df.groupby('cluster').size()
    summary.to_csv(os.path.join(output_dir, 'cluster_averages.csv'))

    with open(os.path.join(output_dir, 'cluster_score.txt'), 'w') as file:
        file.write(f"Silhouette score: {score:.4f}\n")

    df.to_csv(os.path.join(output_dir, 'clustered_data.csv'), index=False)

    for cluster_id, row in summary.iterrows():
        print(
            f"  Group {cluster_id}: "
            f"Patients = {int(row['patient_count'])}, "
            f"Avg Weight Change = {row['weight_change_kg_6m']:.2f} kg"
        )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='baseline_bmi', hue='cluster', data=df, palette='viridis')
    plt.title('Patient Groups by Age and BMI')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_plot.png'))
    plt.close()

    print(f"Clustering results saved in: {output_dir}")
    return df


if __name__ == "__main__":
    run_clustering()
