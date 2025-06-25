import pandas as pd
import numpy as np
import os

INPUT_PATH = 'data/CICIDS2017.csv'
OUTPUT_PATH = 'data/synthetic_zero_day.csv'
FEATURE_OUTPUT = 'data/synthetic_zero_day_features.csv'  # only features, for model testing

def inject_anomalies(df, anomaly_ratio=0.01):
    df = df.copy()
    num_anomalies = int(anomaly_ratio * len(df))

    # Add a label column
    df['label'] = 0

    # Identify numerical columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'label' in numeric_columns:
        numeric_columns.remove('label')

    # Cast all numeric columns to float64 to allow safe noise addition
    df[numeric_columns] = df[numeric_columns].astype(np.float64)

    # Choose random rows to mark as anomalies
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    df.loc[anomaly_indices, 'label'] = 1

    # Apply noise to numeric columns only
    noise = np.random.normal(loc=0.5, scale=0.2, size=(num_anomalies, len(numeric_columns)))
    df.loc[anomaly_indices, numeric_columns] += noise

    return df, numeric_columns

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    df_with_anomalies, feature_columns = inject_anomalies(df)

    # Save full dataset with labels for later evaluation
    df_with_anomalies.to_csv(OUTPUT_PATH, index=False)

    # Save just the feature columns for feeding into model
    df_with_anomalies[feature_columns].to_csv(FEATURE_OUTPUT, index=False)
    print(f"Saved labeled anomalies to {OUTPUT_PATH}")
    print(f"Saved feature-only test set to {FEATURE_OUTPUT}")

if __name__ == '__main__':
    main()