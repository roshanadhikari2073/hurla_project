from hurla_pipeline import run_pipeline
import config

if __name__ == "__main__":
    run_pipeline(config.TRAIN_PATH, config.TEST_PATH)
