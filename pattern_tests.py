import pandas as pd
import os

def run_pattern_tests(input_path='data/merged_data.csv'):
    """
    Checks for specific patterns mentioned in the project requirements.
    """
    print("\n--- TASK 4.5: COMPLEX PATTERN IDENTIFICATION ---")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Simple fix for text columns
    if 'sex' in df.columns:
        df['sex'] = df['sex'].str.strip().str.upper().replace({'FEMALE': 'F', 'MALE': 'M'})
    if 'approach' in df.columns:
        df['approach'] = df['approach'].str.strip().str.lower()

    # Pattern 1: Are patients < 50 more successful in cold months?
    if 'record_created_at' in df.columns and 'age' in df.columns:
        print("Checking Pattern 1: Age < 50 and Seasonality...")
        
        # Convert date to month
        df['month'] = pd.to_datetime(df['record_created_at']).dt.month
        # Cold months: Oct (10), Nov (11), Dec (12), Jan (1), Feb (2), Mar (3)
        df['is_cold'] = df['month'].isin([10, 11, 12, 1, 2, 3])
        
        young_patients = df[df['age'] < 50]
        cold_avg = young_patients[young_patients['is_cold'] == True]['weight_change_kg_6m'].mean()
        warm_avg = young_patients[young_patients['is_cold'] == False]['weight_change_kg_6m'].mean()
        
        print(f"  Avg Weight Change (Cold): {cold_avg:.2f} kg")
        print(f"  Avg Weight Change (Warm): {warm_avg:.2f} kg")
        
        if cold_avg < warm_avg:
            print("  Result: Confirmed. Patients < 50 lose more weight in cold months.")
        else:
            print("  Result: Not confirmed in this dataset.")

    # Pattern 2: Do strict nutritionists have more male patients?
    if 'approach' in df.columns and 'sex' in df.columns:
        print("\nChecking Pattern 2: Strict Nutritionists and Gender...")
        
        strict_group = df[df['approach'] == 'strict']
        other_group = df[df['approach'] != 'strict']
        
        strict_male_pct = (strict_group['sex'] == 'M').mean() * 100
        other_male_pct = (other_group['sex'] == 'M').mean() * 100
        
        print(f"  Male % with Strict Nutritionists: {strict_male_pct:.2f}%")
        print(f"  Male % with Other Nutritionists: {other_male_pct:.2f}%")
        
        if strict_male_pct > other_male_pct:
            print("  Result: Confirmed. Strict nutritionists have more male patients.")
        else:
            print("  Result: Not confirmed in this dataset.")

if __name__ == "__main__":
    run_pattern_tests()
