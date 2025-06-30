import pandas as pd
import numpy as np
import os

# Define input dataset path and output directory for synthetic batches
INPUT_PATH = 'data/CICIDS2017.csv'
OUTPUT_DIR = 'data/gaussian_batches'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# This function injects Gaussian anomalies into a small, selective set of numeric features
def inject_gaussian_anomalies(df, mean, std, anomaly_ratio=0.01, top_n_features=10):
    df = df.copy()
    
    # Ensure all rows are initially labeled as normal (0)
    df['label'] = 0

    # Select only numeric columns (excluding 'label' if already present)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_columns:
        numeric_columns.remove('label')

    # Ensure consistent floating-point type
    df[numeric_columns] = df[numeric_columns].astype('float64')

    # Decide number of rows to turn into anomalies
    num_anomalies = int(anomaly_ratio * len(df))
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)

    # Label those rows as anomalous
    df.loc[anomaly_indices, 'label'] = 1

    # Instead of modifying all numeric features, only alter top N features (greater impact)
    selected_columns = numeric_columns[:top_n_features]  # Top N numeric columns (by order)
    noise = np.random.normal(loc=mean, scale=std, size=(num_anomalies, len(selected_columns)))

    # Inject noise into selected features only for the anomalous rows
    df.loc[anomaly_indices, selected_columns] += noise

    # Return full DataFrame and list of numeric columns for feature-only saving
    return df, numeric_columns

# Main routine to generate 20 batches with varying severity of anomalies
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file {INPUT_PATH} not found.")
        return

    # Read original clean CICIDS dataset
    df = pd.read_csv(INPUT_PATH)
    batch_id = 1

    # Iterate over gradually increasing mean and standard deviation values
    # to simulate worsening anomaly intensity
    for mean in np.linspace(0.3, 1.0, 5):          # 5 noise levels for mean: [0.3, 0.475, ..., 1.0]
        for std in np.linspace(0.3, 0.8, 4):        # 4 levels for std: [0.3, 0.466, ..., 0.8]
            if batch_id > 20:
                break

            # Inject anomalies with current mean and std settings
            df_anom, feature_cols = inject_gaussian_anomalies(df, mean, std)

            # Save both labeled and features-only versions
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