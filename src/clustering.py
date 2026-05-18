import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_clustering(input_path='data/cleaned_data.csv', output_dir='clustering_results'):
    """
    Groups patients into 4 clusters based on their characteristics.
    """
    print("\n--- TASK 4: PATTERN IDENTIFICATION (CLUSTERING) ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    
    # 1. Select features for grouping
    # We use age, bmi, and behavior
    features = ['age', 'baseline_bmi', 'motivation_score', 'mean_adherence_pct']
    X = df[features]
    
    # 2. Run K-Means with 4 groups
    # random_state ensures we get the same result every time
    print("Running K-Means algorithm (k=4)...")
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = model.fit_predict(X)
    
    # 3. Analyze what each group looks like
    summary = df.groupby('cluster')[features + ['weight_change_kg_6m']].mean()
    summary.to_csv(os.path.join(output_dir, 'cluster_averages.csv'))
    print("Cluster averages saved to CSV.")
    
    for i in range(4):
        avg_weight = summary.loc[i, 'weight_change_kg_6m']
        print(f"  Group {i}: Avg Weight Change = {avg_weight:.2f} kg")

    # 4. Create a plot to visualize the groups
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='baseline_bmi', hue='cluster', data=df, palette='viridis')
    plt.title('Patient Groups by Age and BMI')
    plt.savefig(os.path.join(output_dir, 'cluster_plot.png'))
    plt.close()

    print(f"Clustering results saved in: {output_dir}")
    return df

if __name__ == "__main__":
    run_clustering()
