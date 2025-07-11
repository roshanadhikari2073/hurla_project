# config.py ── canonical paths for CICIDS-2018 run
TRAIN_PATH   = "data/processed/CICIDS2018_benign_cleaned.csv"   # now the 77-col clean matrix
MODEL_PATH   = "models/autoencoder_2018.keras"
SCALER_PATH  = "models/minmax_scaler_2018.pkl"
EXPECTED_COL = "models/expected_columns_2018.txt"               # new: the 77 column names
THRESHOLD    = 0.002                                            # starter cut-off; Q-agent will adjust
Q_TABLE_PATH = "models/q_table.json"