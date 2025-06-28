# --------------------------------------
# hurla_pipeline.py (MAJOR CHANGES)
# --------------------------------------
# We import and integrate QLearningAgent.

import numpy as np
import time
import os
import pandas as pd  # Required for label reading

from models.autoencoder import AutoencoderModel
from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate
from tensorflow.keras.models import load_model
from models.q_learning_agent import QLearningAgent  # New: import Q-learning agent

import config

MODEL_PATH = config.MODEL_PATH
Q_TABLE_PATH = config.Q_TABLE_PATH  # New: file path for Q-table persistence


def run_pipeline(train_path, test_path):
    # Load or train autoencoder
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
        ae = AutoencoderModel(model=model)
        threshold = config.THRESHOLD
        print(f"Using threshold from config.py: {threshold}")
    else:
        print("Training new autoencoder model...")
        x_train = preprocess_data(train_path)

        ae = AutoencoderModel(input_dim=x_train.shape[1])
        ae.train(x_train, epochs=10, batch_size=256)

        print("Saving trained model...")
        ae.model.save(MODEL_PATH)

        print("Calculating 95th percentile threshold from reconstruction error...")
        train_recon = ae.model.predict(x_train, verbose=0)
        train_scores = np.mean(np.square(x_train - train_recon), axis=1)
        threshold = np.percentile(train_scores, 95)

        with open("config.py", "r") as f:
            lines = f.readlines()

        with open("config.py", "w") as f:
            for line in lines:
                if line.startswith("THRESHOLD"):
                    f.write(f"THRESHOLD = {threshold}\n")
                else:
                    f.write(line)

        print(f"Threshold saved to config.py: {threshold}")

    print("Loading test data...")
    x_test = preprocess_data(test_path)

    print("Running prediction on test data...")
    start = time.time()
    recons = ae.model.predict(x_test, verbose=1)
    scores = np.mean(np.square(x_test - recons), axis=1)
    latency = (time.time() - start) / len(x_test)

    # New: Initialize Q-learning agent
    agent = QLearningAgent(state_size=10, action_size=3)
    agent.load_q_table(Q_TABLE_PATH)

    print("Using Q-learning agent to adapt threshold...")
    preds = []
    mean_score = np.mean(scores)
    state = min(int(mean_score * 10), 9)  # Discretize score for state mapping
    action = agent.choose_action(state)

    if action == 0:
        threshold *= 0.9
    elif action == 2:
        threshold *= 1.1

    # Log agent's decision and the adjusted threshold
    action_meaning = {0: "DECREASE", 1: "KEEP", 2: "INCREASE"}
    print(f"Q-agent decision: {action_meaning[action]} threshold → {threshold:.10f}")
    
    for s in scores:
        preds.append(1 if s > threshold else 0)

    try:
        labels_df = pd.read_csv(test_path)
        if 'label' in labels_df.columns:
            labels = labels_df['label'].values
        else:
            labels = [0] * len(x_test)
    except:
        labels = [0] * len(x_test)

    metrics = evaluate(preds, labels)
    print(metrics)
    print(f"Avg Latency: {latency * 1000:.2f} ms")

    # New: reward and Q-table update
    accuracy = metrics['Accuracy']
    reward = 1.0 if accuracy > 0.95 else -1.0
    next_state = state  # Simplified assumption
    agent.update(state, action, reward, next_state)
    agent.save_q_table(Q_TABLE_PATH)
    print("Q-table updated and saved.")
