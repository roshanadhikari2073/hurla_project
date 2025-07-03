import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score
from models.q_learning_agent import QLearningAgent

# ===== CONFIGURATION =====
THRESHOLD_MIN = 1e-4
THRESHOLD_MAX = 1e-1

MODEL_PATH = "models/autoencoder_model.keras"
BATCH_DIR = "data/uniform_batches"           # Keep this if batches are reused
SUFFIX = "uniform_replay"                    # Clear and seed-agnostic identifier
LOG_DIR = "logs"
RECON_LOG = os.path.join(LOG_DIR, "reconstruction_errors.csv")

# Log file paths for metrics, rewards, and threshold trajectory
METRICS_LOG = os.path.join(LOG_DIR, f"batch_metrics_{SUFFIX}.csv")
REWARD_LOG = os.path.join(LOG_DIR, f"reward_log_{SUFFIX}.csv")
THRESHOLD_LOG = os.path.join(LOG_DIR, f"threshold_log_{SUFFIX}.csv")
THRESHOLD_TRACKER = os.path.join(LOG_DIR, "last_threshold.txt")  # Shared threshold state

os.makedirs(LOG_DIR, exist_ok=True)

print("Loading trained autoencoder model...")
model = load_model(MODEL_PATH)

# Discover batch files (features only); assume corresponding labeled versions exist
batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*_features.csv")))
print(f"Found {len(batch_files)} batches in '{BATCH_DIR}'")

# Load persisted threshold, or set to None if first-time run
if os.path.exists(THRESHOLD_TRACKER):
    with open(THRESHOLD_TRACKER, 'r') as f:
        last_threshold = float(f.read().strip())
else:
    last_threshold = None

# Initialize Q-agent with three actions: 0 = decrease, 1 = keep, 2 = increase
agent = QLearningAgent(actions=[0, 1, 2])

# Collect logs per batch
metrics_log = []
reward_log = []
threshold_log = []

# Main batch loop
for idx, feature_path in enumerate(batch_files, start=1):
    base_name = os.path.basename(feature_path)
    label_path = feature_path.replace("_features.csv", ".csv")

    if not os.path.exists(label_path):
        print(f"Missing label file for {base_name}, skipping...")
        continue

    print(f"\n=== Evaluating batch {idx}: {base_name} ===")

    # Load features and full data (including labels)
    x_df = pd.read_csv(feature_path)
    full_df = pd.read_csv(label_path)

    if "label" not in full_df.columns:
        print(f"Missing 'label' column in {label_path}, skipping...")
        continue

    x = x_df.values.astype("float64")
    y = full_df["label"].astype(int).values

    # Autoencoder reconstruction
    x_pred = model.predict(x, verbose=0)
    mse = np.mean(np.square(x - x_pred), axis=1)

    # If no prior threshold, use 95th percentile of benign MSEs
    if last_threshold is None:
        benign_mse = mse[y == 0]
        if len(benign_mse) == 0:
            print("No benign samples found for threshold computation, skipping batch.")
            continue
        last_threshold = np.percentile(benign_mse, 95)
        print(f"Computed 95th percentile threshold: {last_threshold:.6f}")
    else:
        print(f"Using persisted threshold: {last_threshold:.6f}")

    # Initial prediction and baseline metrics
    y_pred = (mse > last_threshold).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Construct discrete Q-state and choose action
    state = agent._get_state(precision, recall, f1)
    action = agent.choose_action(state)

    # Adjust threshold based on chosen action
    original_threshold = last_threshold
    if action == 0:
        last_threshold *= 0.5
    elif action == 2:
        last_threshold *= 1.5

    # Enforce threshold boundaries
    last_threshold = np.clip(last_threshold, THRESHOLD_MIN, THRESHOLD_MAX)
    last_threshold = max(last_threshold, 1e-6)

    # Re-evaluate predictions under the new threshold
    new_pred = (mse > last_threshold).astype(int)
    new_precision = precision_score(y, new_pred, zero_division=0)
    new_recall = recall_score(y, new_pred, zero_division=0)
    new_f1 = f1_score(y, new_pred, zero_division=0)
    new_state = agent._get_state(new_precision, new_recall, new_f1)

    # Define reward as change in F1; basic reinforcement signal
    reward = new_f1 - f1
    tp = int(np.sum((y == 1) & (new_pred == 1)))
    fp = int(np.sum((y == 0) & (new_pred == 1)))

    # Update Q-table with observed transition
    agent.update(state, action, reward, new_state)

    # Optional replay update from prior batch memory
    if len(agent.replay) > 10:
        agent.replay_sample()

    # Persist threshold for next iteration
    agent.save_current_threshold(last_threshold)

    # Record batch-level metrics and diagnostics
    metrics_log.append({
        "batch": base_name,
        "threshold": last_threshold,
        "precision": new_precision,
        "recall": new_recall,
        "f1_score": new_f1,
        "total_samples": len(y),
        "anomalies": int(np.sum(y))
    })

    threshold_log.append({
        "batch": base_name,
        "original_threshold": original_threshold,
        "action": ["DECREASE", "KEEP", "INCREASE"][action],
        "new_threshold": last_threshold,
        "mean_mse": float(np.mean(mse))
    })

    reward_log.append({
        "batch": base_name,
        "reward": reward,
        "TP": tp,
        "FP": fp,
        "action": ["DECREASE", "KEEP", "INCREASE"][action],
        "threshold": last_threshold
    })

    # Save reconstruction errors per batch for post-analysis
    recon_df = pd.DataFrame({
        "batch": [base_name] * len(mse),
        "mse": mse
    })
    recon_df.to_csv(RECON_LOG, mode='a', index=False, header=not os.path.exists(RECON_LOG))

# Final log output to disk
pd.DataFrame(metrics_log).to_csv(METRICS_LOG, index=False)
pd.DataFrame(reward_log).to_csv(REWARD_LOG, index=False)
pd.DataFrame(threshold_log).to_csv(THRESHOLD_LOG, index=False)

print(f"\nSaved adaptive batch metrics to {METRICS_LOG}")
print(f"Saved reward log to {REWARD_LOG}")
print(f"Saved threshold adjustments to {THRESHOLD_LOG}")