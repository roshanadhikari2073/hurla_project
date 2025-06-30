import pandas as pd
import numpy as np
import os

INPUT_PATH = 'data/CICIDS2017.csv'
OUTPUT_PATH = 'data/synthetic_zero_day.csv'
FEATURE_OUTPUT = 'data/synthetic_zero_day_features.csv'  # only features, for model testing

def inject_anomalies(df, anomaly_ratio=0.01):
    df = df.copy()
    num_anomalies = int(anomaly_ratio * len(df))

    df['label'] = 0

    # Choose sensitive features with high signal-to-noise impact
    sensitive_cols = [
        "Flow Duration",
        "Total Fwd Packets",
        "Fwd Packet Length Mean"
    ]

    # Ensure all expected features exist
    for col in sensitive_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Convert to float for numerical stability
    df[sensitive_cols] = df[sensitive_cols].astype('float64')

    # Randomly pick rows to corrupt
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    df.loc[anomaly_indices, 'label'] = 1

    # Inject amplified Gaussian noise into sensitive columns only
    for col in sensitive_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()

        # Stronger anomaly signal: centered around +3σ with spread of 1.5σ
        noise = np.random.normal(loc=3 * col_std, scale=1.5 * col_std, size=num_anomalies)
        df.loc[anomaly_indices, col] += noise

    return df, sensitive_cols

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    df_with_anomalies, used_columns = inject_anomalies(df)

    df_with_anomalies.to_csv(OUTPUT_PATH, index=False)
    df_with_anomalies[used_columns].to_csv(FEATURE_OUTPUT, index=False)

    print(f"Saved labeled anomalies to {OUTPUT_PATH}")
    print(f"Saved feature-only test set to {FEATURE_OUTPUT}")
    print(f"Injected strong noise into: {', '.join(used_columns)}")

if __name__ == '__main__':
    main()