import sys
from src.data_integration import integrate_data
from src.preprocessing import clean_data
from src.eda_analysis import run_eda
from src.pattern_tests import run_pattern_tests
from src.clustering import run_clustering
from src.modeling import run_modeling

def main():
    print("====================================================")
    print("      DIET SUCCESS ANALYSIS & PREDICTION SYSTEM")
    print("====================================================")

    # 1. Data Integration 
    df_merged = integrate_data()
    if df_merged is None:
        print("Integration failed. Stopping.")
        return

    # 2. Preprocessing & Normalization 
    df_cleaned = clean_data()
    if df_cleaned is None:
        print("Preprocessing failed. Stopping.")
        return

    # 3. Exploratory Data Analysis 
    run_eda()

    # 4. Complex Pattern Identification / Hypotheses 
    run_pattern_tests()

    # 5. Clustering 
    run_clustering()

    # 6. Prediction Modeling 
    run_modeling()

    print("\n====================================================")
    print("      PROJECT COMPLETED SUCCESSFULLY!")
    print("====================================================")
    print("Check the results in 'data/', 'eda_results/', 'clustering_results/', and 'modeling_results/'.")

if __name__ == "__main__":
    main()
