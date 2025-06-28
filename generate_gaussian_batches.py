import pandas as pd
import numpy as np
import os

INPUT_PATH = 'data/CICIDS2017.csv'
OUTPUT_DIR = 'data/gaussian_batches'  # Directory to hold all output files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# this function applies gaussian noise to selected numeric features
def inject_gaussian_anomalies(df, mean, std, anomaly_ratio=0.01):
    df = df.copy()
    df['label'] = 0  # reset all to normal
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'label' in numeric_columns:
        numeric_columns.remove('label')

    df[numeric_columns] = df[numeric_columns].astype('float64')
    num_anomalies = int(anomaly_ratio * len(df))
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    df.loc[anomaly_indices, 'label'] = 1

    noise = np.random.normal(loc=mean, scale=std, size=(num_anomalies, len(numeric_columns)))
    df.loc[anomaly_indices, numeric_columns] += noise
    return df, numeric_columns

# this is the core loop to generate 20 batches with increasing noise variance
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    batch_id = 1

    # vary mean from 0.1 to 1.0, std from 0.05 to 0.5
    for mean in np.linspace(0.1, 1.0, 5):          # 5 levels of mean
        for std in np.linspace(0.05, 0.5, 4):       # 4 levels of std
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
