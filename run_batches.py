# run_batches.py
import os
import glob
import config
from hurla_pipeline import run_pipeline

# Directory and anomaly type config
BATCH_DIR = "data/gaussian_batches"
ANOMALY_TYPE = "gaussian"
os.environ["ANOMALY_TYPE"] = ANOMALY_TYPE

# Locate all batch files
feature_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*_features.csv")))
print(f"Found {len(feature_files)} batches in '{BATCH_DIR}'")

# Run each batch through the pipeline
for i, test_path in enumerate(feature_files, start=1):
    print(f"\n=== Batch {i}: Running pipeline for {os.path.basename(test_path)} ===")
    run_pipeline(config.TRAIN_PATH, test_path)