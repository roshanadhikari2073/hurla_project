import os
import pandas as pd

BATCH_DIR = "data/gaussian_batches"
batch_files = sorted([f for f in os.listdir(BATCH_DIR) if f.endswith(".csv") and "features" not in f])

print(f"\nFound {len(batch_files)} labeled batch files.\n")

for i, file in enumerate(batch_files, start=1):
    path = os.path.join(BATCH_DIR, file)
    df = pd.read_csv(path)

    if 'label' not in df.columns:
        print(f"Batch {i}: {file} → 'label' column NOT FOUND")
        continue

    pos_count = df['label'].sum()
    total = len(df)
    percent = (pos_count / total) * 100 if total > 0 else 0

    print(f"Batch {i}: {file} → {int(pos_count)} positive labels out of {total} rows ({percent:.4f}%)")
