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

ANOMALY_TYPE = os.environ.get("ANOMALY_TYPE", "default")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

THRESHOLD_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_threshold_log.csv"
METRICS_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_metrics_log.csv"
REWARD_LOG_PATH = f"{LOG_DIR}/{ANOMALY_TYPE}_reward_log.csv"

MODEL_PATH = config.MODEL_PATH

def run_pipeline(train_path, test_path):
    if os.path.exists(MODEL_PATH):
        print("Loading trained model...")
        model = load_model(MODEL_PATH)
        ae = AutoencoderModel(model=model)
    else:
        print("Training new model...")
        x_train = preprocess_data(train_path)
        ae = AutoencoderModel(input_dim=x_train.shape[1])
        ae.train(x_train, epochs=10, batch_size=256)
        ae.model.save(MODEL_PATH)

    print("Loading test data...")
    x_test = preprocess_data(test_path)

    print("Running inference...")
    start = time.time()
    recon = ae.model.predict(x_test, verbose=0)
    scores = np.mean(np.square(x_test - recon), axis=1)
    latency = (time.time() - start) / len(x_test)

    # Load test labels if available
    try:
        df = pd.read_csv(test_path)
        labels = df['label'].values if 'label' in df.columns else [0] * len(scores)
    except:
        labels = [0] * len(scores)

    agent = QLearningAgent(actions=[0, 1, 2])
    threshold = agent.get_last_threshold(default=config.THRESHOLD)
    original_threshold = threshold

    # Make predictions based on threshold
    preds = [1 if s > threshold else 0 for s in scores]

    # Evaluate predictions
    metrics = evaluate(preds, labels)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    f1 = metrics.get("f1_score", 0.0)

    state = agent._get_state(precision, recall, f1)
    action = agent.select_action(state)

    step = 0.5
    if action == 0:
        threshold *= (1 - step)
    elif action == 2:
        threshold *= (1 + step)

    new_preds = [1 if s > threshold else 0 for s in scores]
    new_metrics = evaluate(new_preds, labels)
    new_precision = new_metrics.get("precision", 0.0)
    new_recall = new_metrics.get("recall", 0.0)
    new_f1 = new_metrics.get("f1_score", 0.0)
    new_state = agent._get_state(new_precision, new_recall, new_f1)

    tp = new_metrics.get("TP", 0)
    fp = new_metrics.get("FP", 0)
    reward = tp - fp

    print(f"Q-agent action: {['DECREASE','KEEP','INCREASE'][action]} → threshold {threshold:.6f}")
    print(f"Reward: {reward} | Precision: {new_precision:.4f} | Recall: {new_recall:.4f} | F1: {new_f1:.4f}")

    batch_file = os.path.basename(test_path)
    timestamp = datetime.now().isoformat()

    # Persist threshold and update agent
    agent.update(state, action, reward, new_state)
    agent.save_current_threshold(threshold)

    # Logs
    threshold_log = {
        "timestamp": timestamp,
        "batch_file": batch_file,
        "original_threshold": original_threshold,
        "action": ["DECREASE", "KEEP", "INCREASE"][action],
        "new_threshold": threshold,
        "mean_score": float(np.mean(scores))
    }
    pd.DataFrame([threshold_log]).to_csv(
        THRESHOLD_LOG_PATH, mode='a', index=False, header=not os.path.exists(THRESHOLD_LOG_PATH)
    )

    new_metrics.update({
        "timestamp": timestamp,
        "batch_file": batch_file,
        "latency_ms": latency * 1000
    })
    pd.DataFrame([new_metrics]).to_csv(
        METRICS_LOG_PATH, mode='a', index=False, header=not os.path.exists(METRICS_LOG_PATH)
    )

    reward_log = {
        "timestamp": timestamp,
        "batch_file": batch_file,
        "reward": reward,
        "TP": tp,
        "FP": fp,
        "FPR": new_metrics.get("FPR", 0.0),
        "action": ["DECREASE", "KEEP", "INCREASE"][action],
        "threshold": threshold
    }
    pd.DataFrame([reward_log]).to_csv(
        REWARD_LOG_PATH, mode='a', index=False, header=not os.path.exists(REWARD_LOG_PATH)
    )

    print("Pipeline execution complete.")