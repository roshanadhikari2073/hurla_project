import pandas as pd
import numpy as np
import os

INPUT_PATH = 'data/CICIDS2017.csv'
OUTPUT_DIR = 'data/gaussian_batches'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inject Gaussian anomalies into only a subset of numeric features
def inject_gaussian_anomalies(df, mean, std, anomaly_ratio=0.01, top_n_features=10):
    df = df.copy()
    df['label'] = 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'label' in numeric_columns:
        numeric_columns.remove('label')

    df[numeric_columns] = df[numeric_columns].astype('float64')
    num_anomalies = int(anomaly_ratio * len(df))
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    df.loc[anomaly_indices, 'label'] = 1

    # Limit noise to the first N numeric features
    selected_columns = numeric_columns[:top_n_features]
    noise = np.random.normal(loc=mean, scale=std, size=(num_anomalies, len(selected_columns)))
    df.loc[anomaly_indices, selected_columns] += noise

    return df, numeric_columns

# Generate 20 batches with gradually increasing noise severity
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    batch_id = 1

    # Adjust noise ranges: stronger and more impactful
    for mean in np.linspace(0.3, 1.0, 5):          # 5 levels of mean
        for std in np.linspace(0.3, 0.8, 4):        # 4 levels of std
            if batch_id > 20:
                break
            df_anom, feature_cols = inject_gaussian_anomalies(df, mean, std)
            
            labeled_output = os.path.join(OUTPUT_DIR, f'gaussian_batch_{batch_id:02d}.csv')
            features_only_output = os.path.join(OUTPUT_DIR, f'gaussian_batch_{batch_id:02d}_features.csv')
            
            df_anom.to_csv(labeled_output, index=False)
            df_anom[feature_cols].to_csv(features_only_output, index=False)
            
            print(f"Batch {batch_id:02d}: mean={mean:.2f}, std={std:.2f}")
            print(f"Saved labeled to {labeled_output}")
            print(f"Saved features to {features_only_output}")
            
            batch_id += 1

if __name__ == '__main__':
    main()