# config.py

# Path to training dataset (preprocessed numeric features)
TRAIN_PATH = "data/CICIDS2017_features.csv"

# Path to test dataset (used for manual or fallback testing)
TEST_PATH = "data/synthetic_zero_day_features.csv"

# Default fallback threshold (in case no prior value is saved)
THRESHOLD = 0.002582674299179656

# Trained autoencoder model file
MODEL_PATH = "models/autoencoder_model.keras"

# Updated Q-table JSON path for new adaptive Q-learning agent
Q_TABLE_PATH = "models/q_table.json"

# Optional: Cleaned full dataset (used in batch generation or analysis)
INPUT_FILE = "data/CICIDS2017_full_cleaned.csv"