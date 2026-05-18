import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def run_modeling(input_path='data/cleaned_data.csv', output_dir='modeling_results'):
    """
    Trains models to predict weight loss.
    """
    print("\n--- TASK 5: PREDICTION MODELING ---")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    df = pd.read_csv(input_path)
    
    # 1. Prepare Features (X) and Target (y)
    target = 'weight_change_kg_6m'
    y = df[target]
    
    # Drop target from features
    X = df.drop(columns=[target])
    
    # Convert text columns to numbers (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # 2. Data Split (50% Train, 30% Val, 10% Test, 10% Unused)
    print("Splitting data into Train(50%), Val(30%), Test(10%), and Unused(10%)...")
    
    # Step 1: Separate 10% Unused
    X_working, X_unused, y_working, y_unused = train_test_split(X, y, test_size=0.10, random_state=42)
    
    # Step 2: From the remaining 90%, take 50% for Train
    # (50/90) of 90% is 50%
    X_train, X_temp, y_train, y_temp = train_test_split(X_working, y_working, test_size=(40/90), random_state=42)
    
    # Step 3: From the remaining 40%, take 30% for Val and 10% for Test
    # (10/40) of 40% is 10%
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(10/40), random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")

    # 3. Model 1: Linear Regression (Simple Baseline)
    print("Training Model 1: Linear Regression...")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # 4. Model 2: Decision Tree (Interpretable)
    print("Training Model 2: Decision Tree...")
    model_dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    model_dt.fit(X_train, y_train)
    
    # 5. Model 3: Random Forest (Powerful Ensemble)
    print("Training Model 3: Random Forest...")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    # 6. Compare Models on Validation Set
    models = [("Linear Reg", model_lr), ("Decision Tree", model_dt), ("Random Forest", model_rf)]
    results = []
    
    print("\nResults on Validation Set:")
    for name, model in models:
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        print(f"  {name:15} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
        results.append({'Model': name, 'RMSE': rmse, 'R2': r2})
        
    # Save results to CSV
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    # 7. Visualize the Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(model_dt, feature_names=X.columns.tolist(), filled=True, max_depth=2, fontsize=10)
    plt.title("Simplified Decision Tree Structure")
    plt.savefig(os.path.join(output_dir, 'tree_visualization.png'))
    plt.close()
    
    print(f"Modeling artifacts saved in: {output_dir}")

if __name__ == "__main__":
    run_modeling()
