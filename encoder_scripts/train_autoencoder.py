import os
import pandas as pd
from sklearn.model_selection import train_test_split

from models.autoencoder import AutoencoderModel
from utils.preprocessing import preprocess_data

from config import MODEL_PATH, INPUT_FILE  # Make sure these are correctly pointing

def train_autoencoder():
    print(f"Loading cleaned data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    if 'label' not in df.columns:
        raise ValueError("The dataset must contain a 'label' column.")

    benign_df = df[df['label'] == 0].drop(columns=['label'])

    if benign_df.empty:
        raise ValueError("No benign samples found for training.")

    print(f"Total benign samples loaded: {len(benign_df)}")

    # Optional: if preprocessing logic is centralized
    x = preprocess_data(benign_df)

    # Split for basic validation (not used for early stopping, just to mimic train/test)
    X_train, X_val = train_test_split(x, test_size=0.1, random_state=42)

    print("Initializing autoencoder model...")
    ae = AutoencoderModel(input_dim=X_train.shape[1])

    print("Training autoencoder...")
    ae.train(X_train, epochs=20, batch_size=256)

    print("Saving trained model...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    ae.model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_autoencoder()