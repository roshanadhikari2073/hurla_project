import numpy as np
import time
import os

from models.autoencoder import AutoencoderModel
from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate
from tensorflow.keras.models import load_model

import config

MODEL_PATH = "models/autoencoder_model.keras"

def run_pipeline(train_path, test_path):
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

    preds = [1 if s > threshold else 0 for s in scores]

    # If 'label' column exists in test CSV, load it
    try:
        labels_df = pd.read_csv(test_path)
        if 'label' in labels_df.columns:
            labels = labels_df['label'].values
        else:
            labels = [0] * len(x_test)
    except:
        labels = [0] * len(x_test)

    print(evaluate(preds, labels))
    print(f"Avg Latency: {latency * 1000:.2f} ms")