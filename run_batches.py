import os
import glob
from hurla_pipeline import run_pipeline
import config

# define the parent directory containing all batches (you can change this)
BATCH_DIR = "data/gaussian_batches"

# scan for all feature CSV files ending in _features.csv inside that directory
feature_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*_features.csv")))

# sanity check: print how many batches we found
print(f"Found {len(feature_files)} batches in '{BATCH_DIR}'")

# loop through each batch file and run the main pipeline
for i, test_path in enumerate(feature_files, start=1):
    print(f"\n=== Batch {i}: Running pipeline for {os.path.basename(test_path)} ===")
    run_pipeline(config.TRAIN_PATH, test_path)



