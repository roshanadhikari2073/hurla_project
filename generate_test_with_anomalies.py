"""
Anomaly Batch Generator for Intrusion Detection Experiments

This script generates synthetic anomaly-labeled datasets using three different perturbation styles:
1. Gaussian noise injection
2. Uniform random noise injection
3. Fixed-value feature corruption

Each method creates labeled and normalized batches from the cleaned CICIDS2017 dataset. The anomalies are injected
at a configurable ratio (default 5%), and the result is saved as pairs of CSV files—one with all features and labels,
and one containing features only—for use in unsupervised anomaly detection pipelines.

Batch-specific seeds ensure reproducibility. All output batches are normalized to [0, 1] using MinMax scaling.

Directory Output:
- data/gaussian_batches/
- data/uniform_batches/
- data/fixed_batches/

This generator supports evaluation of the generalization capacity of threshold-based detectors across
distinct anomaly types and perturbation strategies.
"""

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

def inject_uniform_anomalies(df, anomaly_ratio=0.05, low=-3.0, high=3.0, seed=None):
    """
    Injects uniform noise-based anomalies into a subset of the dataframe.
    """
    np.random.seed(seed)
    df = df.copy()
    num_rows = len(df)
    num_anomalies = int(num_rows * anomaly_ratio)

    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)

    for col in feature_columns:
        noise = np.random.uniform(low=low, high=high, size=num_anomalies)
        df.loc[anomaly_indices, col] += noise

    df["label"] = 0
    df.loc[anomaly_indices, "label"] = 1
    return df

def inject_fixed_value_anomalies(df, anomaly_ratio=0.05, fixed_value=9999, seed=None):
    """
    Injects fixed-value anomalies into a subset of the dataframe.
    """
    np.random.seed(seed)
    df = df.copy()
    num_rows = len(df)
    num_anomalies = int(num_rows * anomaly_ratio)

    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)

    for col in feature_columns:
        df.loc[anomaly_indices, col] = fixed_value

    df["label"] = 0
    df.loc[anomaly_indices, "label"] = 1
    return df

def inject_gaussian_anomalies(df, anomaly_ratio=0.05, std_dev=2.5, seed=None):
    """
    Injects Gaussian noise-based anomalies into a subset of the dataframe.
    """
    np.random.seed(seed)
    df = df.copy()
    num_rows = len(df)
    num_anomalies = int(num_rows * anomaly_ratio)

    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)

    for col in feature_columns:
        noise = np.random.normal(loc=0, scale=std_dev, size=num_anomalies)
        df.loc[anomaly_indices, col] += noise

    df["label"] = 0
    df.loc[anomaly_indices, "label"] = 1

    return df

def create_batches(df, batch_size, output_dir, inject_func, inject_kwargs, seed=None):
    """
    Splits the dataset into batches, injects anomalies, and writes labeled and feature-only CSVs per batch.
    """
    num_batches = int(np.ceil(len(df) / batch_size))
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end].copy()

        kwargs = inject_kwargs.copy()
        kwargs["seed"] = seed + i if seed is not None else None
        batch_df = inject_func(batch_df, **kwargs)

        feature_cols = batch_df.drop(columns=["label"]).columns
        label_path = os.path.join(output_dir, f"batch_{i+1:02d}.csv")
        feature_path = os.path.join(output_dir, f"batch_{i+1:02d}_features.csv")

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

    print("Creating Gaussian anomaly batches...")
    create_batches(df, BATCH_SIZE, "data/gaussian_batches", inject_gaussian_anomalies, {"anomaly_ratio": ANOMALY_RATIO, "std_dev": STD_DEV}, SEED)

    print("Creating Uniform anomaly batches...")
    create_batches(df, BATCH_SIZE, "data/uniform_batches", inject_uniform_anomalies, {"anomaly_ratio": ANOMALY_RATIO, "low": -2.0, "high": 2.0}, SEED)

    print("Creating Fixed-value anomaly batches...")
    create_batches(df, BATCH_SIZE, "data/fixed_batches", inject_fixed_value_anomalies, {"anomaly_ratio": ANOMALY_RATIO, "fixed_value": 9999}, SEED)

    print("All anomaly batches generated and saved.")