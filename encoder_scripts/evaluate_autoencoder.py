import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

MODEL_PATH = "models/autoencoder_model.keras"
EVAL_DATA_PATH = "data/CICIDS2017_full_clean_fixed.csv"
RECON_ERRORS_OUTPUT = "logs/reconstruction_errors.csv"

def evaluate_autoencoder():
    print("Loading evaluation data...")
    df = pd.read_csv(EVAL_DATA_PATH)

    if 'label' not in df.columns:
        raise ValueError("Evaluation data must include a 'label' column.")

    x = df.drop(columns=['label']).values
    y = df['label'].astype(int).values

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    print("Performing reconstruction...")
    x_pred = model.predict(x, verbose=0)

    mse = np.mean(np.square(x - x_pred), axis=1)
    result_df = pd.DataFrame({'reconstruction_error': mse, 'label': y})
    result_df.to_csv(RECON_ERRORS_OUTPUT, index=False)

    print("Saved reconstruction errors to", RECON_ERRORS_OUTPUT)

    # Example classification using static threshold
    threshold = np.percentile(mse[y == 0], 95)  # Use 95th percentile of benign samples
    y_pred = (mse > threshold).astype(int)

    print(f"\nUsing threshold: {threshold:.6f}")
    print(classification_report(y, y_pred, target_names=["BENIGN", "ATTACK"]))

if __name__ == "__main__":
    if not os.path.exists(EVAL_DATA_PATH):
        print(f"Input file {EVAL_DATA_PATH} not found.")
    elif not os.path.exists(MODEL_PATH):
        print(f"Trained model not found at {MODEL_PATH}")
    else:
        evaluate_autoencoder()
