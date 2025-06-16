import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    # Drop non-numeric or irrelevant fields
    for col in ['ip.src', 'ip.dst', 'frame.time_epoch']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df)

    return X

# Alias to match import in hurla_pipeline.py
preprocess_data = load_and_preprocess
