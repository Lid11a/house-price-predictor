from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DESCRIPTION_PATH = RAW_DATA_DIR / "data_description.txt"
DEFAULT_TRAIN_PATH = RAW_DATA_DIR / "train.csv"
DEFAULT_TEST_PATH = RAW_DATA_DIR / "test.csv"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
DEFAULT_PREDICTIONS_PATH = ARTIFACTS_DIR / "submission.csv"
