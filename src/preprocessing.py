import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def clean_data(input_path='data/merged_data.csv', output_dir='data'):
    """
    Cleans the data: removes duplicates, handles missing values, and fixes labels.
    """
    print("\n--- TASK 2: DATA CLEANING & PREPROCESSING ---")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return None
        
    df = pd.read_csv(input_path)
    
    # 1. Remove Duplicate Rows
    df = df.drop_duplicates()
    
    # 2. Remove ID columns (not useful for prediction)
    columns_to_drop = ['patient_id', 'diet_id', 'nutritionist_id', 'bmi_redundant', 'experience_years']
    # Only drop columns that exist in the file
    existing_cols = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    print(f"Removed redundant columns: {existing_cols}")

    # 3. Fix labels (Sex and Approach)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].str.strip().str.upper()
        df['sex'] = df['sex'].replace({'FEMALE': 'F', 'MALE': 'M'})
        print(f"Standardized sex labels: {df['sex'].unique()}")
        
    if 'approach' in df.columns:
        df['approach'] = df['approach'].str.strip().str.lower()
        print(f"Standardized nutritionist approach labels: {df['approach'].unique()}")

    # 4. Handle Missing Values
    # If the target (weight change) is missing, we must drop those rows
    target = 'weight_change_kg_6m'
    if target in df.columns:
        df = df.dropna(subset=[target])

    # Fill other numerical missing values with Median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # Fill categorical missing values with Mode
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print("Missing values handled (Median for numbers, Mode for text).")

    # 5. Handle Outliers (Capping at 3 Standard Deviations)
    for col in num_cols:
        if col != target:
            limit_upper = df[col].mean() + 3 * df[col].std()
            limit_lower = df[col].mean() - 3 * df[col].std()
            df[col] = np.clip(df[col], limit_lower, limit_upper)
    print("Outliers handled via capping.")

    # 6. Normalization (Requirement Task 2)
    scaler = StandardScaler()
    # Scale all numerical columns except the target
    cols_to_scale = [c for c in num_cols if c != target]
    
    if len(cols_to_scale) > 0:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"Normalization applied to {len(cols_to_scale)} features.")
        
    output_path = os.path.join(output_dir, 'cleaned_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    clean_data()
