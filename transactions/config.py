import os

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TRAIN_FILE_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")

MODEL_PATH = "check.model"