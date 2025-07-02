import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT_DATA = "data/CICIDS2017_full_clean_fixed.csv"
OUTPUT_DIR = "data/gaussian_batches"
BATCH_SIZE = 20000
ANOMALY_RATIO = 0.05
STD_DEV = 2.5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def inject_gaussian_anomalies(df, anomaly_ratio=0.05, std_dev=2.5, seed=None):
    np.random.seed(seed)
    df = df.copy()
    num_rows = len(df)
    num_anomalies = int(num_rows * anomaly_ratio)

    # Identify numeric features
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Choose anomaly rows
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)

    # Inject Gaussian noise across all numeric features
    for col in feature_columns:
        noise = np.random.normal(loc=0, scale=std_dev, size=num_anomalies)
        df.loc[anomaly_indices, col] += noise

    # Label data
    df["label"] = 0
    df.loc[anomaly_indices, "label"] = 1

    return df

def create_batches(df, batch_size, output_dir, anomaly_ratio, std_dev, seed=None):
    num_batches = int(np.ceil(len(df) / batch_size))

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end].copy()

        batch_df = inject_gaussian_anomalies(batch_df, anomaly_ratio=anomaly_ratio, std_dev=std_dev, seed=seed + i if seed else None)

        feature_cols = batch_df.drop(columns=["label"]).columns
        label_path = os.path.join(output_dir, f"gaussian_batch_{i+1:02d}.csv")
        feature_path = os.path.join(output_dir, f"gaussian_batch_{i+1:02d}_features.csv")

        batch_df.to_csv(label_path, index=False)
        batch_df[feature_cols].to_csv(feature_path, index=False)

        print(f"Saved batch {i+1:02d} to {label_path} and {feature_path}")

if __name__ == "__main__":
    print("Loading input data...")
    df = pd.read_csv(INPUT_DATA)

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    print("Normalizing features to [0, 1] range...")
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    print("Creating batches with synthetic anomalies...")
    create_batches(df, BATCH_SIZE, OUTPUT_DIR, ANOMALY_RATIO, STD_DEV, SEED)

    print("All batches generated and saved.")