import os
import subprocess
import pandas as pd

# ------------------ SEED CONFIGURATION ------------------

# List of seeds for reproducible batch generation and evaluation
seeds = [7, 42, 99, 123, 2025]
all_logs = []

# ------------------ MULTI-SEED PIPELINE EXECUTION ------------------

for seed in seeds:
    print(f"\n=== Running experiment for seed {seed} ===")

    # Step 1 — Generate 24 Gaussian noise-injected batches for current seed
    subprocess.run([
        "python", "generate_gaussian_batches.py",
        "--seed", str(seed),
        "--output_dir", f"data/uniform_batches_seed{seed}"
    ], check=True)

    # Step 2 — Run Q-learning based adaptive thresholding on generated batches
    subprocess.run([
        "python", "run_batches.py",
        "--seed", str(seed)
    ], check=True)

    # Step 3 — Load metrics log and tag with current seed for later consolidation
    log_path = f"logs/batch_metrics_uniform_replay_seed{seed}.csv"
    df = pd.read_csv(log_path)
    df["seed"] = seed
    all_logs.append(df)

# ------------------ MERGE LOGS INTO LONG-FORMAT METRICS FILE ------------------

# Combine batch-level metrics across all seeds into a single DataFrame
merged_df = pd.concat(all_logs, ignore_index=True)
merged_df.to_csv("logs/merged_metrics_multiseed.csv", index=False)

print("\n[✓] All seeds processed.")
print("    Merged log saved to → logs/merged_metrics_multiseed.csv")