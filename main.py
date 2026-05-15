import sys
from src.data_integration import integrate_data
from src.preprocessing import clean_data

def main():
    print("====================================================")
    print("      DIET SUCCESS ANALYSIS & PREDICTION SYSTEM")
    print("====================================================")

    # 1. Data Integration (Task 1)
    df_merged = integrate_data()
    if df_merged is None:
        print("Integration failed. Stopping.")
        return

    # 2. Preprocessing & Normalization (Task 2)
    df_cleaned = clean_data()
    if df_cleaned is None:
        print("Preprocessing failed. Stopping.")
        return

    print("\n====================================================")
    print("      TASKS 1 & 2 COMPLETED SUCCESSFULLY!")
    print("====================================================")
    print("Check the results in 'data/'.")

if __name__ == "__main__":
    main()
