import sys
from src.data_integration import integrate_data
from src.preprocessing import clean_data
from src.eda_analysis import run_eda
from src.pattern_tests import run_pattern_tests

def main():
    print("====================================================")
    print("      DIET SUCCESS ANALYSIS & PREDICTION SYSTEM")
    print("====================================================")

    # 1. Data Integration (Task 1) — endrick
    df_merged = integrate_data()
    if df_merged is None:
        print("Integration failed. Stopping.")
        return

    # 2. Preprocessing & Normalization (Task 2) — endrick
    df_cleaned = clean_data()
    if df_cleaned is None:
        print("Preprocessing failed. Stopping.")
        return

    # 3. Exploratory Data Analysis (Task 3) — elkin
    run_eda()

    # 4. Complex Pattern Identification / Hypotheses (Task 4) — elkin
    run_pattern_tests()

    print("\n====================================================")
    print("      TASKS 1-4 COMPLETED SUCCESSFULLY!")
    print("====================================================")
    print("Check the results in 'data/' and 'eda_results/'.")

if __name__ == "__main__":
    main()
