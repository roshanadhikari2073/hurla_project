# hurla_pipeline.py
# ─────────────────────────────────────────────────────────────────────
# 77-feature online inference with:
#   • global Min–Max scaling
#   • per-batch constant-dim suppression
#   • robust-Z anomaly scores
#   • dynamic 99.9-percentile “anchor” for daily drift
#   • EMA smoothing toward yesterday’s threshold
#   • Q-agent ±10% nudge
#   • precision-floor loop to enforce TARGET_PRECISION
#
#  Steps:
#   1.  Load & harmonise CSV → 77 numeric features + label
#   2.  Scale with the global MinMaxScaler
#   3.  Reconstruct via Autoencoder → per-sample MSE
#   4.  Drop any feature-dims that are all zero this batch
#   5.  Compute robust-Z scores
#   6.  Anchor = score 99.9-percentile; EMA blend with last_thr
#   7.  Q-agent nudge; ramp threshold until precision ≥ TARGET_PRECISION
#   8.  Log metrics, reward & new threshold
# ─────────────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model

from models.autoencoder      import AutoencoderModel
from models.q_learning_agent import QLearningAgent
from utils.evaluation        import evaluate
import config

# ── singleton artefacts: load once per process ────────────────────────
MODEL   = AutoencoderModel(model=load_model(config.MODEL_PATH))
SCALER  = joblib.load(config.SCALER_PATH)
AGENT   = QLearningAgent(actions=[0, 1, 2])

# ── expected features + metadata headers ──────────────────────────────
with open(config.EXPECTED_COL) as f:
    COLS = [c.strip() for c in f if c.strip()]

META = {c.lower() for c in (
    "flow id","timestamp","src ip","dst ip",
    "source ip","destination ip","src port","dst port","label"
)}

# ── helper: align incoming DataFrame → exact 77-col numeric matrix ─────
def align(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in df.columns if c.lower() not in META]].copy()
    df = (df
          .apply(pd.to_numeric, errors="coerce")
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0))
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[COLS].astype("float32")

# ── helper: robust-Z normalisation ────────────────────────────────────
def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    return (x - med) / mad

# ── helper: safe confusion-matrix wrapper ────────────────────────────
def safe_eval(pred: np.ndarray, lab: np.ndarray) -> dict:
    if len(np.unique(np.r_[pred, lab])) == 1:
        # all one class → zero out metrics
        return dict(TP=0, FP=0,
                    TN=len(lab) if lab[0] == 0 else 0,
                    FN=0, precision=0.0, recall=0.0, f1=0.0)
    return evaluate(pred, lab)

# ── logging convenience ───────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def _log(fname: str, row: dict):
    path = os.path.join(LOG_DIR, fname)
    pd.DataFrame([row]).to_csv(
        path, mode="a", index=False,
        header=not os.path.exists(path)
    )

# ── main pipeline entrypoint ──────────────────────────────────────────
def run_pipeline(_unused, csv_path: str,
                 min_prec: float = config.TARGET_PRECISION) -> None:
    """
    Process one CSV batch:
      - load & label
      - feature-align → scale
      - AE reconstruction → errors
      - suppress all-zero dims → robust-Z scores
      - dynamic anchor → EMA → Q-nudge → precision-floor
      - log metrics, reward, threshold
    """
    # 1) load & harmonise label
    df = (pd.read_csv(csv_path, low_memory=False)
          .rename(columns=str.strip))
    df["label"] = (
        df.get("Label", df.get("label","BENIGN"))
          .astype(str).str.upper()
          .map(lambda s: 0 if s=="BENIGN" else 1)
    )

    # 2) align + scale
    feats = align(df)
    if feats.empty:
        print(f"[Skip] {csv_path}: no numeric rows")
        return
    X = SCALER.transform(feats)
    y = df.loc[feats.index, "label"].to_numpy(int)

    # 3) reconstruct & compute per-sample MSE
    recon = np.vstack([
        MODEL.model.predict(X[i:i+1024], verbose=0)
        for i in tqdm(range(0, len(X), 1024), desc="Reconstruct")
    ])

    # 4) drop any feature dims that are all zero this batch
    mask      = (X != 0).any(axis=0)
    X_use     = X[:, mask]
    recon_use = recon[:, mask]

    # 5) mean-squared errors + robust-Z
    errors = ((X_use - recon_use) ** 2).mean(axis=1)
    scores = robust_z(errors)

    # 6) dynamic anchor + EMA smoothing toward last threshold
    anchor    = np.percentile(scores, config.ANCHOR_PCTL)
    last_thr  = AGENT.get_last_threshold(config.THRESHOLD)
    alpha     = getattr(config, "THR_SMOOTH", 0.7)
    start_thr = np.clip(anchor,
                        config.MIN_THR_ABS, config.MAX_THR_ABS)
    thr       = alpha * last_thr + (1 - alpha) * start_thr

    # 7) first-pass metrics
    learn = y.sum() > 0
    p1    = (scores > thr).astype(int) if learn else np.zeros_like(scores, int)
    m1    = safe_eval(p1, y)
    TP, FP, FN = m1["TP"], m1["FP"], m1["FN"]
    prec  = TP / (TP + FP) if TP + FP else 0.0
    rec1  = TP / (TP + FN) if TP + FN else 0.0
    f1    = 2 * prec * rec1 / (prec + rec1) if prec + rec1 else 0.0

    # 8) Q-learning nudge: ±10% bump or drop
    if learn:
        s0  = AGENT._get_state(prec, rec1, f1)
        act = AGENT.choose_action(s0)
        thr = np.clip(
            thr * config.DWN_FAC if act == 0 else
            thr * config.UP_FAC  if act == 2 else thr,
            config.MIN_THR_ABS, config.MAX_THR_ABS
        )
    else:
        act = 1  # keep

    # 9) precision-floor loop: lift until precision ≥ target
    while True:
        p2    = (scores > thr).astype(int)
        m2    = safe_eval(p2, y)
        TP2, FP2, FN2 = m2["TP"], m2["FP"], m2["FN"]
        prec2 = TP2 / (TP2 + FP2) if TP2 + FP2 else 0.0
        rec2  = TP2 / (TP2 + FN2) if TP2 + FN2 else 0.0
        if prec2 >= min_prec or thr >= config.MAX_THR_ABS:
            break
        thr *= config.UP_FAC

    # 10) compute reward (FP penalty + optional recall bonus)
    reward = (
        (TP2 * 2)
        - config.REWARD_FP_PENALTY * FP2
        + getattr(config, "REWARD_REC_FACTOR", 0.0) * rec2
    )

    # 11) update agent & persist threshold
    if learn:
        s1 = AGENT._get_state(prec2, rec2,
                              2 * prec2 * rec2 / (prec2 + rec2 or 1))
        AGENT.update(s0, act, reward, s1)
        AGENT.save_current_threshold(thr)

    # 12) log & display
    fn = os.path.basename(csv_path)
    _log("threshold_log.csv",
         dict(file=fn, orig_thr=last_thr, new_thr=thr,
              anchor=anchor, action=act))
    _log("metrics_log.csv",
         dict(file=fn, **m2, precision=prec2, recall=rec2))
    _log("reward_log.csv",
         dict(file=fn, reward=reward))

    print(f"[Done] {fn} | thr {thr:.2f} | "
          f"P {prec:.3f}->{prec2:.3f} R {rec1:.3f}->{rec2:.3f}")