![CI](https://github.com/lid11a/house-price-predictor/actions/workflows/ci.yml/badge.svg)

# House Price Prediction

---

A regression machine learning project focused on predicting **house sale prices**
from structured tabular data.

**Languages:** [English](README.md) | [Русский](README_RU.md)

---

## Project snapshot

A project covering the full machine learning workflow for house price prediction — from exploratory analysis 
and model comparison to a reproducible training pipeline and a FastAPI-based inference service.

- Structured notebook with EDA, statistical analysis, feature relationship analysis, and model comparison
- Final engineering implementation built around a regularized linear model (Lasso) selected from experiments
- Shared preprocessing pipeline for training and inference, along with structured logging
- FastAPI service for online predictions
- Automated tests with pytest and CI via GitHub Actions

---

## Quick start

This repository contains **two parts**:

1. **Notebook / research layer** — exploratory analysis, experiments, and final model selection  
2. **Code implementation layer** — reproducible training script, prediction script, API, and tests

The notebook part can be explored separately by opening:

```text
notebooks/house-price-predictor.ipynb
```

The code below demonstrates how to work with the implementation part of the project:

- train the final model;
- generate predictions for the Kaggle test set;
- run the FastAPI service locally for online predictions.

**Prerequisites:**

- Python 3.9+ (tested on Python 3.12)
- Kaggle API token for dataset download

### 1) Clone the repository

```bash
git clone https://github.com/lid11a/house-price-predictor.git
cd house-price-predictor
```

### 2) Configure Kaggle API token

- **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
- **Linux / macOS:** `~/.kaggle/kaggle.json`

For Linux/macOS, set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 3) Create a virtual environment and install dependencies

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4) Train the model

```bash
python src/train.py
```

### 5) Generate predictions for the test.csv test dataset

```bash
python src/test.py
```

### 6) Run the API locally

```bash
uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Table of contents

- [Project overview](#project-overview)
- [Exploratory analysis](#exploratory-analysis)
- [Code implementation](#code-implementation)
- [Project structure](#project-structure)
- [Data](#data)
- [Running the project](#running-the-project)
- [Technologies used](#technologies-used)
- [License](#license)

---

## Project overview

This project demonstrates a complete workflow for a tabular regression problem:
from exploratory data analysis and statistical reasoning to comparative modeling and
a reproducible implementation for training and inference.

The project is intentionally split into two complementary parts.

### 1) Exploratory / notebook part

Implemented in:

```text
notebooks/house-price-predictor.ipynb
```

This part contains the analytical work used to understand the dataset and compare model families.
It includes:

- target distribution analysis and justification of log-transformation;
- analysis of numerical and categorical features;
- Pearson / Spearman correlation analysis;
- eta-squared style analysis for categorical relationships;
- comparison of multiple regression model families;
- interpretation of model behavior and final model selection.

Several model families are explored and compared in the notebook, including:

- linear models;
- regularized linear models;
- polynomial regression;
- distance-based models;
- tree-based models and ensembles;
- gradient boosting models.

The experiments lead to a final practical conclusion: 
a regularized linear approach provides the best trade-off between generalization performance, 
simplicity, and stability for this dataset.
That final choice is what is implemented in the code layer.

So the notebook shows **the full research process and model comparison**,
while the codebase implements **the final selected workflow**.

### 2) Code / engineering part

Implemented in `src/` and `tests/`.

This part translates the notebook conclusions into a smaller reproducible ML package.
It includes:

- reproducible dataset download from Kaggle;
- training script for the final model;
- saved artifact with model and metadata;
- batch prediction script for the competition test set;
- FastAPI inference service;
- feature documentation endpoints;
- automated tests and CI.

During training, the following files are saved:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

The artifact stores the fitted model and metadata such as the target column and scikit-learn version.

#### API service

A small FastAPI app is implemented in:

```text
src/api/main.py
```

Available endpoints:

- `GET /health` — service liveness check
- `GET /schema` — machine-readable input schema
- `GET /cheatsheet` — markdown guide for features and categorical values
- `POST /predict` — online prediction for one or more records

The API reuses the same saved artifact as the batch prediction script.

#### Logging

Structured logging is configured for script-based workflows.

Log files:

- `logs/train.log`
- `logs/predict_batch.log`

#### Testing and quality control

The project includes automated tests for key parts of the implementation.

Current tests cover:

- API behavior
- core modeling helpers
- scikit-learn compatibility patch

Tests are executed through `pytest` and can be run locally or in CI.

---

## Project structure

```text
house-price-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI configuration (GitHub Actions): dependency installation and test execution
├── notebooks/
│   └── house-price-predictor.ipynb # EDA, data analysis, experiments, and final model selection
├── src/
│   ├── api/
│   │   ├── __init__.py             # API package
│   │   └── main.py                 # FastAPI application: /health, /schema, /cheatsheet, /predict
│   ├── house_price_predictor/
│   │   ├── __init__.py             # package with the core project logic
│   │   ├── config.py               # configuration and paths (data, artifacts, logs, etc.)
│   │   ├── data.py                 # utilities for loading and reading CSV data
│   │   ├── download.py             # dataset download from Kaggle via CLI
│   │   ├── feature_schema.py       # generation of the input data schema and API cheatsheet
│   │   ├── logging_config.py       # logging setup (files + stderr)
│   │   ├── modeling.py             # pipeline construction and helper modeling functions
│   │   ├── predict.py              # model loading and predictions (for API and batch use)
│   │   └── sklearn_compat.py       # compatibility patch for scikit-learn versions
│   ├── train.py                    # CLI script for model training and artifact saving
│   └── test.py                     # CLI script for batch predictions on test.csv (not pytest)
├── tests/
│   ├── test_api.py                 # API tests (endpoints and responses)
│   ├── test_modeling.py            # tests for modeling logic
│   └── test_sklearn_compat.py      # tests for the scikit-learn compatibility patch
├── artifacts/                      # saved model artifacts (model.joblib, metrics.json), generated locally
├── data/                           # raw data (train.csv, test.csv), downloaded locally
├── logs/                           # training and batch inference logs, generated locally
├── .gitignore                      # files and directories excluded from Git
├── LICENSE                         # project license
├── pytest.ini                      # pytest configuration (paths, settings)
├── README.md                       # project description (English version)
├── README_RU.md                    # project description (Russian version)
└── requirements.txt                # list of dependencies
```

---

## Data

The project uses the Kaggle competition dataset:

```text
house-prices-advanced-regression-techniques
```

The repository does not store the raw competition data.
Instead, data is downloaded locally through the Kaggle CLI.

Default local paths:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/data_description.txt`

If `train.csv` and `test.csv` already exist locally, download is skipped unless forced.

---

## Running the project

All commands below are executed from the repository root.

### 1) Install dependencies

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run tests

```bash
pytest
```

### 3) Train the final model

```bash
python src/train.py
```

Optional:

```bash
python src/train.py --force-download
```

Artifacts produced:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

### 4) Generate predictions for the Kaggle test set

```bash
python src/test.py
```

Optional:

```bash
python src/test.py --force-download
```

Output:

- `artifacts/submission.csv`

### 5) Run the API

```bash
uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000
```

Useful URLs:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/schema`
- `http://127.0.0.1:8000/cheatsheet`
- `http://127.0.0.1:8000/docs`

### 6) CI (GitHub Actions)

The GitHub Actions workflow is located at:

```text
.github/workflows/ci.yml
```

The pipeline runs automatically on:

- `push`
- `pull_request`

The current CI steps are:

- checkout repository
- set up Python 3.11
- install dependencies
- run `pytest`

---

## Technologies used

### Notebook / exploratory layer

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- catboost
- Jupyter Notebook

### Code / implementation layer

- Python
- pandas
- numpy
- scikit-learn
- joblib
- FastAPI
- Uvicorn
- pytest
- httpx
- Kaggle CLI
- GitHub Actions

---

## License

This project is licensed under the MIT License.
