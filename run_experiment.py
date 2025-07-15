# run_experiment.py

import os
from hurla_pipeline import run_pipeline
import config

if __name__ == "__main__":
    # Optional: set the anomaly type for log naming
    os.environ["ANOMALY_TYPE"] = "gaussian"  # or 'default', 'uniform', etc.

    # Trigger the full pipeline run
    run_pipeline(config.TRAIN_PATH, config.TEST_PATH)