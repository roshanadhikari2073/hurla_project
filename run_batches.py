import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration section
MODEL_PATH = "models/autoencoder_model.keras"  # Path to trained autoencoder model
BATCH_DIR = "data/gaussian_batches"            # Directory containing feature + label batch files
LOG_PATH = "logs/batch_metrics_log.csv"        # Where final metrics log will be stored
THRESHOLD_TRACKER = "logs/last_threshold.txt"  # Used to store/load the latest computed threshold
os.makedirs("logs", exist_ok=True)             # Ensure logs directory exists

# Load trained model
print("Loading trained autoencoder model...")
model = load_model(MODEL_PATH)

# Locate all synthetic batch feature files
batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*_features.csv")))
print(f"Found {len(batch_files)} batches in '{BATCH_DIR}'")

# Read threshold from file if it exists
if os.path.exists(THRESHOLD_TRACKER):
    with open(THRESHOLD_TRACKER, 'r') as f:
        last_threshold = float(f.read().strip())
else:
    last_threshold = None

# Metrics collection
metrics_log = []

# Evaluate each batch
for idx, feature_path in enumerate(batch_files, start=1):
    base_name = os.path.basename(feature_path)
    label_path = feature_path.replace("_features.csv", ".csv")

    if not os.path.exists(label_path):
        print(f"Missing label file for {base_name}, skipping...")
        continue

    print(f"\n=== Evaluating batch {idx}: {base_name} ===")

    x_df = pd.read_csv(feature_path)
    full_df = pd.read_csv(label_path)

    if 'label' not in full_df.columns:
        print(f"Missing 'label' column in {label_path}, skipping...")
        continue

    x = x_df.values.astype("float64")
    y = full_df['label'].astype(int).values

    # Predict reconstruction
    x_pred = model.predict(x, verbose=0)
    mse = np.mean(np.square(x - x_pred), axis=1)

    # Initialize threshold using benign sample distribution
    if last_threshold is None:
        benign_mse = mse[y == 0]
        if len(benign_mse) == 0:
            print("No benign samples found for threshold computation, skipping batch.")
            continue
        last_threshold = np.percentile(benign_mse, 95)
        with open(THRESHOLD_TRACKER, 'w') as f:
            f.write(f"{last_threshold:.6f}")
        print(f"Computed 95th percentile threshold: {last_threshold:.6f}")
    else:
        print(f"Using persisted threshold: {last_threshold:.6f}")

    # Apply threshold for classification
    y_pred = (mse > last_threshold).astype(int)

    # Evaluate predictions
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    # Log metrics
    metrics_log.append({
        "batch": base_name,
        "threshold": last_threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_samples": len(y),
        "anomalies": int(np.sum(y))
    })

# Write out the entire batch summary
log_df = pd.DataFrame(metrics_log)
log_df.to_csv(LOG_PATH, index=False)
print(f"\nSaved batch evaluation metrics to {LOG_PATH}")