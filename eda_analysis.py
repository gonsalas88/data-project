import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(input_path='data/cleaned_data.csv', output_dir='eda_results'):
    """
    Analyzes the data and creates plots.
    """
    print("\n--- TASK 3: EXPLORATORY DATA ANALYSIS ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    
    # 1. Summary Statistics
    # Save a CSV with mean, min, max, etc.
    df.describe().to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    print("Summary statistics saved as CSV.")

    # 2. Correlation Matrix
    # Shows which variables move together
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numerical_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    print("Correlation matrix plot saved.")

    # 3. Categorical Analysis
    target = 'weight_change_kg_6m'
    
    # Sex vs Weight Change
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='sex', y=target, data=df)
    plt.title('Weight Change by Sex')
    plt.savefig(os.path.join(output_dir, 'success_by_sex.png'))
    plt.close()

    # Diet Type vs Weight Change
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diet_type', y=target, data=df)
    plt.xticks(rotation=45)
    plt.title('Weight Change by Diet Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_by_diet_type.png'))
    plt.close()

    # 4. Outlier Visualization
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numerical_df)
    plt.xticks(rotation=90)
    plt.title('Numerical Distributions and Outliers')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_check.png'))
    plt.close()

    print(f"All plots saved in the folder: {output_dir}")

if __name__ == "__main__":
    run_eda()
