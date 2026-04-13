from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


DEFAULT_KAGGLE_COMPETITION = "house-prices-advanced-regression-techniques"


def _run_kaggle_download(competition: str, output_dir: Path) -> Path:
    # Run the Kaggle CLI and download the competition archive into output_dir.
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{competition}.zip"

    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(output_dir),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return zip_path


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    # Extract a downloaded Kaggle zip into the raw data directory.
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Kaggle archive was not created: {zip_path}. "
            "Check Kaggle CLI auth and competition access."
        )
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def ensure_kaggle_house_prices_data(
    train_path: Path,
    test_path: Path,
    *,
    force_download: bool = False,
    competition: str = DEFAULT_KAGGLE_COMPETITION,
) -> None:
    # Download and unzip the dataset when train/test are missing or force_download is set.
    if not force_download and train_path.exists() and test_path.exists():
        return

    output_dir = train_path.parent
    try:
        zip_path = _run_kaggle_download(competition=competition, output_dir=output_dir)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI is not installed or unavailable in PATH. "
            "Install dependencies and ensure 'kaggle' command works."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Kaggle download failed: {message}") from exc

    _extract_zip(zip_path=zip_path, output_dir=output_dir)
