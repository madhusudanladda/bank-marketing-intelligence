# src/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

CSV_FILENAME = "bank-additional-full.csv"

# Verified active direct CSV mirror
DIRECT_CSV_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/bank-additional-full.csv"

MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.35