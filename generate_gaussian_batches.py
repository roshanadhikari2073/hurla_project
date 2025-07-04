import os
import argparse
import numpy as np
import pandas as pd

# ------------------ ARGUMENTS AND PATH SETUP ------------------

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True,
                    help="Random seed to ensure reproducible Gaussian noise injection")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Output directory to store the generated Gaussian anomaly batches")
args = parser.parse_args()

SEED = args.seed
OUTPUT_DIR = args.output_dir
INPUT_PATH = "data/CICIDS2017_full_clean.csv"  # Clean base dataset used for injection

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ GAUSSIAN ANOMALY INJECTION FUNCTION ------------------

def inject_gaussian_anomalies(df, mean, std, anomaly_ratio=0.05, top_n_features=10):
    """
    Injects synthetic anomalies into a clean dataframe by applying Gaussian noise
    to selected numeric features. Anomalies are labeled as '1', rest remain '0'.
    """
    df = df.copy()
    df['label'] = 0

    # Select only numeric columns for noise injection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_columns:
        numeric_columns.remove('label')
    df[numeric_columns] = df[numeric_columns].astype('float64')

    # Randomly select a subset of rows as anomalies
    num_anomalies = int(anomaly_ratio * len(df))
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    df.loc[anomaly_indices, 'label'] = 1

    # Apply Gaussian noise to top N numeric columns
    selected_columns = numeric_columns[:top_n_features]
    noise = np.random.normal(loc=mean, scale=std, size=(num_anomalies, len(selected_columns)))
    df.loc[anomaly_indices, selected_columns] += noise

    return df, numeric_columns

# ------------------ MAIN EXECUTION FLOW ------------------

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input file '{INPUT_PATH}' not found.")
        return

    df = pd.read_csv(INPUT_PATH)
    batch_id = 1

    # Sweep over a grid of Gaussian parameters (mean × std)
    for mean in np.linspace(0.3, 1.0, 5):
        for std in np.linspace(0.3, 0.8, 4):
            if batch_id > 24:
                break

            # Inject anomalies for current Gaussian configuration
            df_anom, feature_cols = inject_gaussian_anomalies(df, mean, std)

            # Save full batch (with labels) and feature-only batch (for inference)
            full_path    = os.path.join(OUTPUT_DIR, f"gaussian_batch_{batch_id:02d}.csv")
            feature_path = os.path.join(OUTPUT_DIR, f"gaussian_batch_{batch_id:02d}_features.csv")
            df_anom.to_csv(full_path, index=False)
            df_anom[feature_cols].to_csv(feature_path, index=False)

            print(f"[Seed {SEED}] Batch {batch_id:02d} | mean={mean:.2f}, std={std:.2f}")
            print(f"  └─ Saved full batch:     {full_path}")
            print(f"  └─ Saved feature batch:  {feature_path}")

            batch_id += 1

# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    main()