# hurla_pipeline.py ── robust, column-aligned inference
import os, joblib, numpy as np, pandas as pd
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras.models import load_model
from models.autoencoder      import AutoencoderModel
from models.q_learning_agent import QLearningAgent
from utils.evaluation         import evaluate
import config

# ─── once-per-process artefact loading ───────────────────────────────
MODEL = AutoencoderModel(model=load_model(config.MODEL_PATH))
SCALER = joblib.load(config.SCALER_PATH)
AGENT  = QLearningAgent(actions=[0,1,2])

with open(config.EXPECTED_COL) as f:          # training-time header
    COLS = [c.strip() for c in f if c.strip()]

META = {c.lower() for c in (
        "flow id","timestamp","src ip","dst ip",
        "source ip","destination ip","src port","dst port","label")}

# ─── utility: force every frame into the 77-col slot ─────────────────
def align(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in df.columns if c.lower() not in META]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.)
    for c in COLS:
        if c not in df.columns:
            df[c] = 0.
    return df[COLS].astype("float32")

def robust_z(x: np.ndarray) -> np.ndarray:
    m = np.median(x); d = np.median(np.abs(x-m)) or 1e-9
    return (x-m)/d

def safe_eval(p: np.ndarray, y: np.ndarray) -> dict:
    if len(np.unique(np.r_[p,y])) == 1:
        return {"TP":0,"FP":0,"TN":len(y) if y[0]==0 else 0,"FN":0,
                "precision":0.0,"recall":0.0,"f1":0.0}
    return evaluate(p,y)

# ─── paths for run-time logs ─────────────────────────────────────────
LOG = "logs"; os.makedirs(LOG, exist_ok=True)
def _log(fname,row):
    path=f"{LOG}/{fname}"; pd.DataFrame([row]).to_csv(path,mode="a",index=False,
                                header=not os.path.exists(path))

STEP   = 0.25                         # additive threshold tweak
TH_MIN,TH_MAX = 0.10,5.0

# ─── main entry point ────────────────────────────────────────────────
def run_pipeline(_unused, csv, min_prec=0.70):
    raw = pd.read_csv(csv, low_memory=False).rename(columns=str.strip)
    raw["label"] = raw.get("Label", raw.get("label","BENIGN")).astype(str).str.upper()\
                    .map(lambda s: 0 if s=="BENIGN" else 1)

    feats = align(raw)                                   # 77 columns, float32
    if feats.empty:
        print(f"[Skip] {csv}: no numeric rows."); return

    y   = raw.loc[feats.index,"label"].to_numpy(int)
    X   = SCALER.transform(feats)
    rec = np.vstack([MODEL.model.predict(X[i:i+1024],verbose=0)
                     for i in tqdm(range(0,len(X),1024),desc="Reconstruct")])
    scores = robust_z(((X-rec)**2).mean(1))

    thr0 = AGENT.get_last_threshold(config.THRESHOLD)
    p1   = (scores>thr0).astype(int)
    m1   = safe_eval(p1,y); TP,FP,FN=m1["TP"],m1["FP"],m1["FN"]
    prec = TP/(TP+FP) if TP+FP else 0.; rec1 = TP/(TP+FN) if TP+FN else 0.
    f1   = 2*prec*rec1/(prec+rec1) if prec+rec1 else 0.

    if y.sum():
        s0 = AGENT._get_state(prec,rec1,f1)
        act= AGENT.choose_action(s0)
        thr = max(TH_MIN,min(TH_MAX,thr0-STEP if act==0 else thr0+STEP if act==2 else thr0))
    else:
        act,thr = 1,thr0

    p2   = (scores>thr).astype(int)
    m2   = safe_eval(p2,y); TP2,FP2,FN2=m2["TP"],m2["FP"],m2["FN"]
    prec2= TP2/(TP2+FP2) if TP2+FP2 else 0.; rec2=TP2/(TP2+FN2) if TP2+FN2 else 0.
    reward=TP2-FP2*(2 if prec2<min_prec else 1)

    if y.sum():
        s1=AGENT._get_state(prec2,rec2,2*prec2*rec2/(prec2+rec2 or 1))
        AGENT.update(s0,act,reward,s1); AGENT.save_current_threshold(thr)

    ts=os.path.basename(csv); _log("threshold_log.csv",
      {"file":ts,"orig_thr":thr0,"new_thr":thr,"act":act})
    _log("metrics_log.csv",
      {"file":ts,**m2,"precision":prec2,"recall":rec2})
    _log("reward_log.csv",
      {"file":ts,"reward":reward})

    print(f"[Done]{ts} | thr {thr:.2f} | P {prec:.3f}->{prec2:.3f} R {rec1:.3f}->{rec2:.3f}")