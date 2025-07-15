import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Drop any rows with nulls
    df.dropna(inplace=True)

    # Remove non-numeric or irrelevant columns if they exist
    for col in ['ip.src', 'ip.dst', 'frame.time_epoch']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Convert all remaining values to numeric and coerce invalid entries to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace inf/-inf with NaN and drop those rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Apply MinMax normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df)

    return X

# Alias used in hurla_pipeline.py
preprocess_data = load_and_preprocess