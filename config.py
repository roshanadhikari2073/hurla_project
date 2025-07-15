# config.py
# Central hyper-parameter and path definitions

# ── data & model paths ───────────────────────────────────────────────
TRAIN_PATH     = "data/processed/CICIDS2018_benign_cleaned.csv"
TEST_PATH      = "data/raw/CICIDS2018"
MODEL_PATH     = "models/autoencoder_2018.keras"
SCALER_PATH    = "models/minmax_scaler_2018.pkl"
EXPECTED_COL   = "models/expected_columns_2018.txt"

# ── RL persistence ─────────────────────────────────────────────────────
THRESHOLD      = 1.00                       # cold-start
Q_TABLE_PATH   = "logs/q_table.json"        # no longer used

# ── tuning hyper-params ───────────────────────────────────────────────
TARGET_PRECISION   = 0.95                   # enforced post-nudge
MIN_THR_ABS, MAX_THR_ABS = 0.50, 8.00       # allowable threshold bounds
DWN_FAC, UP_FAC    = 0.90, 1.15             # ±10% Q-agent tweak
REWARD_FP_PENALTY  = 20                     # cost per false positive

# ── new knobs for anchor & smoothing ─────────────────────────────────
ANCHOR_PCTL        = 99.9                   # robust-Z percentile
THR_SMOOTH         = 0.70                   # EMA weight on previous thr
REWARD_REC_FACTOR  = 0.50                   # bonus × recall in reward