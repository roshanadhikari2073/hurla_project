# --------------------------------------
# hurla_pipeline.py (FULL PIPELINE WITH LOGGING)
# --------------------------------------

import numpy as np
import time
import os
import pandas as pd
from datetime import datetime

from models.autoencoder import AutoencoderModel
from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate
from tensorflow.keras.models import load_model
from models.q_learning_agent import QLearningAgent

import config

# Set up environment-driven anomaly naming
ANOMALY_TYPE = os.environ.get("ANOMALY_TYPE", "default")
LOG_DIR = "logs"
Q_VALUES_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_q_values_log.csv"
THRESHOLD_TRACKER_FILE = f"{LOG_DIR}/{ANOMALY_TYPE}_last_threshold.txt"
THRESHOLD_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_threshold_log.csv"
METRICS_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_metrics_log.csv"
METRICS_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_metrics_log.csv"
Q_VALUES_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_q_values_log.csv"

MODEL_PATH = config.MODEL_PATH
Q_TABLE_PATH = config.Q_TABLE_PATH

def run_pipeline(train_path, test_path):
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load model or train it
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
        ae = AutoencoderModel(model=model)

        try:
            with open(THRESHOLD_TRACKER_FILE, 'r') as f:
                threshold = float(f.read().strip())
            print(f"Using last threshold from {THRESHOLD_TRACKER_FILE}: {threshold}")
        except FileNotFoundError:
            threshold = config.THRESHOLD
            print(f"No prior threshold found, using default from config.py: {threshold}")
    else:
        print("Training new autoencoder model...")
        x_train = preprocess_data(train_path)

        ae = AutoencoderModel(input_dim=x_train.shape[1])
        ae.train(x_train, epochs=10, batch_size=256)
        ae.model.save(MODEL_PATH)

        print("Calculating 95th percentile threshold...")
        recon = ae.model.predict(x_train, verbose=0)
        scores = np.mean(np.square(x_train - recon), axis=1)
        threshold = np.percentile(scores, 95)

        with open("config.py", "r") as f:
            lines = f.readlines()
        with open("config.py", "w") as f:
            for line in lines:
                if line.startswith("THRESHOLD"):
                    f.write(f"THRESHOLD = {threshold}\n")
                else:
                    f.write(line)

    print("Loading test data...")
    x_test = preprocess_data(test_path)

    print("Running inference on test data...")
    start = time.time()
    recon = ae.model.predict(x_test, verbose=1)
    scores = np.mean(np.square(x_test - recon), axis=1)
    latency = (time.time() - start) / len(x_test)

    agent = QLearningAgent(state_size=10, action_size=3)
    agent.load_q_table(Q_TABLE_PATH)

    mean_score = np.mean(scores)
    state = min(int(mean_score * 10), 9)
    action = agent.choose_action(state)
    original_threshold = threshold

    # Log Q-values for current state
    q_values = agent.q_table[state]
    q_log = {
        "timestamp": datetime.now().isoformat(),
        "batch_file": os.path.basename(test_path),
        "state": state,
        "chosen_action": action
    }
    for i, val in enumerate(q_values):
        q_log[f"q_value_action_{i}"] = val

    pd.DataFrame([q_log]).to_csv(
        Q_VALUES_LOG_PATH, mode='a', index=False, header=not os.path.exists(Q_VALUES_LOG_PATH)
    )

    # Adjust threshold based on Q-agent action
    if action == 0:
        threshold *= 0.9
    elif action == 2:
        threshold *= 1.1

    action_meaning = {0: "DECREASE", 1: "KEEP", 2: "INCREASE"}
    print(f"Q-agent decision: {action_meaning[action]} threshold → {threshold:.10f}")

    # Save new threshold
    with open(THRESHOLD_TRACKER_FILE, 'w') as f:
        f.write(str(threshold))

    # Log threshold change
    threshold_log = {
        "timestamp": datetime.now().isoformat(),
        "batch_file": os.path.basename(test_path),
        "original_threshold": original_threshold,
        "action": action_meaning[action],
        "new_threshold": threshold,
        "mean_score": mean_score
    }
    pd.DataFrame([threshold_log]).to_csv(
        THRESHOLD_LOG_PATH, mode='a', index=False, header=not os.path.exists(THRESHOLD_LOG_PATH)
    )

    preds = [1 if s > threshold else 0 for s in scores]

    try:
        labels_df = pd.read_csv(test_path)
        labels = labels_df['label'].values if 'label' in labels_df.columns else [0] * len(x_test)
    except Exception as e:
        print(f"Warning: Failed to load labels from {test_path}, defaulting to 0s. Error: {e}")
        labels = [0] * len(x_test)

    raw_metrics = evaluate(preds, labels)
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "batch_file": os.path.basename(test_path),
        "Accuracy": float(raw_metrics.get("Accuracy", 0)),
        "F1": float(raw_metrics.get("F1", 0)),
        "FPR": float(raw_metrics.get("FPR", 0)),
        "latency_ms": float(latency * 1000)
    }

    print(metrics)

    os.makedirs(os.path.dirname(METRICS_LOG_PATH), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        METRICS_LOG_PATH, mode='a', index=False, header=not os.path.exists(METRICS_LOG_PATH)
    )

    reward = 1.0 if metrics['Accuracy'] > 0.95 else -1.0
    next_state = state
    agent.update(state, action, reward, next_state)
    agent.save_q_table(Q_TABLE_PATH)
    print("Q-table updated and saved.")