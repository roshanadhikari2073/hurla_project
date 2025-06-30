import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/CICIDS2017_full_labeled_fixed.csv"
CLEAN_OUTPUT = "data/CICIDS2017_full_clean.csv"
FEATURE_ONLY_OUTPUT = "data/CICIDS2017_full_features.csv"

def preprocess_cicids2017(path):
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df.drop(columns=[col for col in df.columns if "Unnamed" in col], inplace=True)

    # Ensure all columns except label are coerced to numeric
    for col in df.columns:
        if col.strip().lower() != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify label column
    label_col = None
    for col in df.columns:
        if col.strip().lower() == 'label':
            label_col = col
            break

    if not label_col:
        print("No label column found. Adding default 'label' column with zeros.")
        df['label'] = 0
        label_col = 'label'
    else:
        try:
            df['label'] = df[label_col].astype(int)
        except:
            print("Warning: Non-integer labels detected. Falling back to string-based mapping.")
            df['label'] = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

    # Re-identify numeric columns after coercion
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' not in numeric_cols:
        numeric_cols.append('label')

    df_numeric = df[numeric_cols].copy()
    print(f"Retained {len(numeric_cols)} numeric columns.")

    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric.fillna(0, inplace=True)

    # Ensure label exists
    if 'label' not in df_numeric.columns:
        print("Label column missing in numeric DataFrame. Inserting manually.")
        df_numeric['label'] = df['label']

    feature_cols = [col for col in df_numeric.columns if col != 'label']
    if not feature_cols:
        raise ValueError("No numeric features found for scaling. Cannot proceed.")

    x = df_numeric[feature_cols].astype("float64")
    y = df_numeric['label'].astype("int")

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=feature_cols)

    df_final = x_scaled_df.copy()
    df_final['label'] = y.values

    print("Saving cleaned and normalized dataset...")
    df_final.to_csv(CLEAN_OUTPUT, index=False)
    x_scaled_df.to_csv(FEATURE_ONLY_OUTPUT, index=False)

    print(f"Done. Clean file saved to {CLEAN_OUTPUT}")
    print(f"Feature-only file saved to {FEATURE_ONLY_OUTPUT}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
    else:
        preprocess_cicids2017(INPUT_FILE)