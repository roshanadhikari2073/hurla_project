import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

DATA_PATH = "data/CICIDS2017_full_clean.csv"
FEATURE_PATH = "data/CICIDS2017_full_features.csv"
MODEL_OUTPUT = "models/autoencoder_model.keras"

def load_benign_data(path):
    print(f"Loading cleaned dataset from {path}...")
    df = pd.read_csv(path)

    if 'label' not in df.columns:
        raise ValueError("The input dataset does not contain a 'label' column.")

    benign_df = df[df['label'] == 0].drop(columns=['label'])
    print(f"Loaded {len(benign_df)} benign samples with {benign_df.shape[1]} features.")
    return benign_df

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def train_autoencoder():
    print("Loading feature data...")
    df = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {len(df)} samples with {df.shape[1]} features each.")

    X_train, X_val = train_test_split(df.values, test_size=0.1, random_state=42)

    print("Building autoencoder model...")
    model = build_autoencoder(X_train.shape[1])

    print("Starting training...")
    history = model.fit(
        X_train,
        X_train,
        epochs=20,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val, X_val),
        verbose=1
    )

    print("Training complete. Saving model...")
    model.save(MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")
    
if __name__ == "__main__":
    train_autoencoder()
