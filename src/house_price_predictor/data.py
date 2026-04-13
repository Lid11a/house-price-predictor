from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    # Create the parent directory for a file path if it does not exist.
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    # Read a CSV from disk or raise FileNotFoundError.
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)
