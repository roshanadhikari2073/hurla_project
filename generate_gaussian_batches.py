import pandas as pd
import numpy as np
import os

INPUT_PATH = 'data/CICIDS2017_full_clean.csv'
OUTPUT_DIR = 'data/gaussian_batches'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inject_gaussian_anomalies(df, mean, std, anomaly_ratio=0.01, top_n_features=10):
    df = df.copy()
    df['label'] = 0

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns.remove('label')
    df[numeric_columns] = df[numeric_columns].astype('float64')

    num_anomalies = int(anomaly_ratio * len(df))
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    df.loc[anomaly_indices, 'label'] = 1

    selected_columns = numeric_columns[:top_n_features]
    noise = np.random.normal(loc=mean, scale=std, size=(num_anomalies, len(selected_columns)))
    df.loc[anomaly_indices, selected_columns] += noise

    return df, numeric_columns

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    batch_id = 1

    for mean in np.linspace(0.3, 1.0, 5):
        for std in np.linspace(0.3, 0.8, 4):
            if batch_id > 20:
                break

            df_anom, feature_cols = inject_gaussian_anomalies(df, mean, std)
            full_output = os.path.join(OUTPUT_DIR, f'gaussian_batch_{batch_id:02d}.csv')
            feature_output = os.path.join(OUTPUT_DIR, f'gaussian_batch_{batch_id:02d}_features.csv')

            df_anom.to_csv(full_output, index=False)
            df_anom[feature_cols].to_csv(feature_output, index=False)

            print(f"Batch {batch_id:02d}: mean={mean:.2f}, std={std:.2f}")
            print(f"Saved: {full_output}")
            print(f"Saved: {feature_output}")
            batch_id += 1

if __name__ == '__main__':
    main()