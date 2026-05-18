import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree


def evaluate_model(model, X, y):
    preds = model.predict(X)
    return {
        'RMSE': np.sqrt(mean_squared_error(y, preds)),
        'MAE': mean_absolute_error(y, preds),
        'R2': r2_score(y, preds),
    }


def save_feature_importance(model, feature_names, output_dir):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    top_features = importance.head(10).sort_values('importance')
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()


def run_modeling(input_path='data/cleaned_data.csv', output_dir='modeling_results'):
    """
    Trains simple supervised models to predict weight change.
    """
    print("\n--- TASK 5: PREDICTION MODELING ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)

    target = 'weight_change_kg_6m'
    y = df[target]
    X = df.drop(columns=[target])

    # Use month instead of the full date string.
    if 'record_created_at' in X.columns:
        X['record_month'] = pd.to_datetime(X['record_created_at'], errors='coerce').dt.month
        X = X.drop(columns=['record_created_at'])

    # Remove IDs because they do not describe patient behavior or diet quality.
    id_columns = ['program_id']
    X = X.drop(columns=[col for col in id_columns if col in X.columns])

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(X.median(numeric_only=True))

    print("Splitting data into Train(50%), Val(30%), Test(10%), and Unused(10%)...")
    X_working, X_unused, y_working, y_unused = train_test_split(
        X, y, test_size=0.10, random_state=42
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_working, y_working, test_size=(40 / 90), random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(10 / 40), random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")

    print("Training Linear Regression...")
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    print("Choosing the best Decision Tree depth using the validation set...")
    best_tree = None
    best_depth = None
    best_rmse = np.inf

    for depth in [3, 5, 7]:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        val_rmse = evaluate_model(tree, X_val, y_val)['RMSE']

        if val_rmse < best_rmse:
            best_tree = tree
            best_depth = depth
            best_rmse = val_rmse

    print(f"Best Decision Tree depth: {best_depth}")

    print("Training Random Forest...")
    forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_model.fit(X_train, y_train)

    models = [
        ('Linear Regression', linear_model),
        (f'Decision Tree depth={best_depth}', best_tree),
        ('Random Forest', forest_model),
    ]

    results = []
    for name, model in models:
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)

        print(
            f"  {name}: "
            f"Val RMSE = {val_metrics['RMSE']:.4f}, "
            f"Test RMSE = {test_metrics['RMSE']:.4f}, "
            f"Test R2 = {test_metrics['R2']:.4f}"
        )

        results.append({
            'Model': name,
            'Val RMSE': val_metrics['RMSE'],
            'Val MAE': val_metrics['MAE'],
            'Val R2': val_metrics['R2'],
            'Test RMSE': test_metrics['RMSE'],
            'Test MAE': test_metrics['MAE'],
            'Test R2': test_metrics['R2'],
        })

    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    save_feature_importance(forest_model, X.columns, output_dir)

    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, feature_names=X.columns.tolist(), filled=True, max_depth=2, fontsize=10)
    plt.title("Simplified Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tree_visualization.png'))
    plt.close()

    # Simple success probability model.
    # Success means losing at least 5 kg in 6 months.
    success_threshold = -5
    y_train_success = (y_train <= success_threshold).astype(int)
    y_val_success = (y_val <= success_threshold).astype(int)
    y_test_success = (y_test <= success_threshold).astype(int)

    print("Training simple success probability model...")
    success_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    success_model.fit(X_train, y_train_success)

    val_success_preds = success_model.predict(X_val)
    test_success_preds = success_model.predict(X_test)
    test_success_probs = success_model.predict_proba(X_test)[:, 1]

    success_results = pd.DataFrame([{
        'Model': 'Decision Tree Classifier',
        'Success Definition': 'weight_change_kg_6m <= -5',
        'Val Accuracy': accuracy_score(y_val_success, val_success_preds),
        'Val F1': f1_score(y_val_success, val_success_preds),
        'Test Accuracy': accuracy_score(y_test_success, test_success_preds),
        'Test F1': f1_score(y_test_success, test_success_preds),
    }])
    success_results.to_csv(os.path.join(output_dir, 'success_model_results.csv'), index=False)

    probability_examples = pd.DataFrame({
        'actual_weight_change_kg_6m': y_test.values,
        'actual_success': y_test_success.values,
        'predicted_success_probability': test_success_probs,
    })
    probability_examples.head(20).to_csv(
        os.path.join(output_dir, 'success_probability_examples.csv'),
        index=False
    )

    print(f"Modeling artifacts saved in: {output_dir}")


if __name__ == "__main__":
    run_modeling()
