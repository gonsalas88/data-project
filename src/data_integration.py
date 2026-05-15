import pandas as pd
import os

def integrate_data(data_dir='data'):
    """
    Integrates the 4 CSV files (patients, diets, nutritionists, outcomes) into a single dataset.
    """
    print("\n--- TASK 1: DATA INTEGRATION ---")
    print(f"Loading data from: {data_dir}...")
    
    # Load files
    try:
        patients = pd.read_csv(os.path.join(data_dir, 'patients.csv'))
        diets = pd.read_csv(os.path.join(data_dir, 'diets.csv'))
        nutritionists = pd.read_csv(os.path.join(data_dir, 'nutritionists.csv'))
        outcomes = pd.read_csv(os.path.join(data_dir, 'outcomes.csv'))
        print("Success: All 4 source CSV files loaded.")
    except Exception as e:
        print(f"Error: Could not find or read CSV files. {e}")
        return None

    # Merge Outcomes with Patients
    # patient_id is the common column
    merged_df = pd.merge(outcomes, patients, on='patient_id', how='left')
    
    # Merge with Diets
    # diet_id is the common column
    merged_df = pd.merge(merged_df, diets, on='diet_id', how='left')
    
    # Merge with Nutritionists
    # nutritionist_id is the common column
    merged_df = pd.merge(merged_df, nutritionists, on='nutritionist_id', how='left')

    print(f"Integration complete. New dataset size: {merged_df.shape}")
    
    # Save the merged dataset
    output_path = os.path.join(data_dir, 'merged_data.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"File saved to: {output_path}")
    
    return merged_df

if __name__ == "__main__":
    integrate_data()
