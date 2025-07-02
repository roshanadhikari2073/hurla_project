import os
import glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

from models.q_learning_agent import QLearningAgent

THRESHOLD_MIN = 1e-4
THRESHOLD_MAX = 1e-1

MODEL_PATH = "models/autoencoder_model.keras"
# BATCH_DIR = "data/gaussian_batches"
BATCH_DIR = "data/uniform_batches"
# BATCH_DIR = "data/fixed_batches"
LOG_DIR = "logs"

METRICS_LOG = os.path.join(LOG_DIR, "batch_metrics_log.csv")
REWARD_LOG = os.path.join(LOG_DIR, "gaussian_reward_log.csv")
THRESHOLD_LOG = os.path.join(LOG_DIR, "gaussian_threshold_log.csv")
THRESHOLD_TRACKER = os.path.join(LOG_DIR, "last_threshold.txt")

os.makedirs(LOG_DIR, exist_ok=True)

print("Loading trained autoencoder model...")
model = load_model(MODEL_PATH)

batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "*_features.csv")))
print(f"Found {len(batch_files)} batches in '{BATCH_DIR}'")

if os.path.exists(THRESHOLD_TRACKER):
    with open(THRESHOLD_TRACKER, 'r') as f:
        last_threshold = float(f.read().strip())
else:
    last_threshold = None

agent = QLearningAgent(actions=[0, 1, 2])

metrics_log = []
reward_log = []
threshold_log = []

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

    x_pred = model.predict(x, verbose=0)
    mse = np.mean(np.square(x - x_pred), axis=1)

    if last_threshold is None:
        benign_mse = mse[y == 0]
        if len(benign_mse) == 0:
            print("No benign samples found for threshold computation, skipping batch.")
            continue
        last_threshold = np.percentile(benign_mse, 95)
        print(f"Computed 95th percentile threshold: {last_threshold:.6f}")
    else:
        print(f"Using persisted threshold: {last_threshold:.6f}")

    y_pred = (mse > last_threshold).astype(int)

    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    state = agent._get_state(precision, recall, f1)
    action = agent.choose_action(state)

    original_threshold = last_threshold
    if action == 0:
        last_threshold *= 0.5
    elif action == 2:
        last_threshold *= 1.5
        
    last_threshold = np.clip(last_threshold, THRESHOLD_MIN, THRESHOLD_MAX)
        
    # Ensure threshold doesn't go below a practical floor
    MIN_THRESHOLD = 1e-6
    last_threshold = max(last_threshold, MIN_THRESHOLD)

    new_pred = (mse > last_threshold).astype(int)
    new_precision = precision_score(y, new_pred, zero_division=0)
    new_recall = recall_score(y, new_pred, zero_division=0)
    new_f1 = f1_score(y, new_pred, zero_division=0)
    new_state = agent._get_state(new_precision, new_recall, new_f1)

    prev_f1 = f1  # F1 before threshold update
    reward = new_f1 - prev_f1  # ΔF1 as reward
    tp = int(np.sum((y == 1) & (new_pred == 1)))
    fp = int(np.sum((y == 0) & (new_pred == 1)))

    agent.update(state, action, reward, new_state)
    agent.save_current_threshold(last_threshold)

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

pd.DataFrame(metrics_log).to_csv(METRICS_LOG, index=False)
pd.DataFrame(reward_log).to_csv(REWARD_LOG, index=False)
pd.DataFrame(threshold_log).to_csv(THRESHOLD_LOG, index=False)

print(f"\nSaved adaptive batch metrics to {METRICS_LOG}")
print(f"Saved reward log to {REWARD_LOG}")
print(f"Saved threshold adjustments to {THRESHOLD_LOG}")