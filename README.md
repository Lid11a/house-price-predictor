# House Price Prediction

---

A regression modeling project focused on **predicting house prices**
using structured tabular data.

The project emphasizes **rigorous data analysis**, **systematic comparison of model families**,
and an **evidence-based selection of the final solution**.

**Languages:** [English](README.md) | [Русский](README_RU.md)

---

## Table of contents

- [Project overview](#project-overview)
- [Data analysis and experiments](#data-analysis-and-experiments)
- [Project structure](#project-structure)
- [Project setup and execution](#project-setup-and-execution)
- [Technologies used](#technologies-used)
- [License](#license)

---

## Project overview

This project demonstrates a complete analytical workflow for tabular data:
from exploratory analysis and hypothesis validation to comparative model evaluation and final conclusions.

Within the project, the following tasks were performed:

- analysis of the target distribution and justification of log-transformation of house prices;
- investigation of numerical and categorical features using Pearson / Spearman correlations and eta-squared statistics;
- comparison of multiple model families using consistent, model-oriented preprocessing pipelines, including:
    - linear models (baseline and regularized); 
    - polynomial regression; 
    - distance-based models; 
    - tree-based models;
- selection of the final solution with an optimal balance between generalization performance and interpretability.

---

## Data analysis and experiments

All analytical work is implemented in the notebook: `notebooks/house_price_predictor.ipynb`.

### Exploratory and statistical analysis

The following key steps were carried out prior to the modeling stage:

- analysis of data structure, feature types, and missing values in the training and test datasets;
- examination of the target distribution and selection of a log-transformation consistent with the RMSLE metric;
- analysis of numerical feature distributions and their relationships with the target variable;
- comparison of linear and monotonic dependencies for numerical features;
- assessment of multicollinearity among numerical features and its implications for model selection;
- analysis of categorical features, including category distributions and 
  quantitative evaluation of their association with the target variable.

### Modeling

A systematic comparison of several families of regression models was conducted using consistent, 
model-oriented preprocessing pipelines and a common evaluation metric (RMSLE).

The following model groups were evaluated:

- Linear models: OLS as a baseline, as well as Ridge, Lasso, and Elastic Net to stabilize linear modeling 
  under high dimensionality and strong feature correlation.
- Polynomial models: linear models with polynomial feature expansion, combined with PCA and regularization.
- Distance-based models: KNN and SVR to assess the applicability of similarity-based approaches.
- Trees and ensembles: Decision Tree, Random Forest, XGBoost, LightGBM, and CatBoost.

Hyperparameter tuning was performed for all models using cross-validation.
Model performance is reported in the original price scale after inverse log-transformation.

### Key findings

- Regularized linear models achieve the best performance on the hold-out set,
  indicating a dominant linear signal in the data.
- Lasso regression yields the lowest RMSLE by effectively suppressing noise
in a high-dimensional one-hot encoded feature space. 
- Gradient boosting models (CatBoost, LightGBM, XGBoost) demonstrate comparable performance
but do not provide a measurable improvement over Lasso.
- Polynomial and distance-based models degrade in performance as feature dimensionality increases.

---

## Project structure

```
house-price-predictor/
│
├── notebooks/                           # Data analysis, experiments, and conclusions
│    └── house-price-predictor.ipynb
├── .gitignore                           # Git ignore rules
├── README.md                            # Project description (EN)
├── README_RU.md                         # Project description (RU)
├── requirements.txt                     # Project dependencies
└── LICENSE                              # Project license
```

---

## Project setup and execution

### Requirements

- Python 3.9+ (tested with Python 3.12);
- Jupyter Notebook

### Installation and execution

#### 1) Clone the repository

```
git clone https://github.com/Lid11a/house-price-predictor
cd house-price-predictor
```

#### 2) Kaggle API token setup (one-time)

The dataset is automatically downloaded using the Kaggle API.

- **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`
- **Linux / macOS:** `~/.kaggle/kaggle.json`

For Linux / macOS, file permissions must be set as follows:

```
 chmod 600 ~/.kaggle/kaggle.json
```

#### 3) Create and activate a virtual environment, then install dependencies

**Windows (PowerShell)**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4) Run the analysis

Launch Jupyter Notebook or Jupyter Lab and execute the notebook
`notebooks/house-price-predictor.ipynb` sequentially from top to bottom.

---

## Technologies used

The project leverages a set of tools covering the full tabular data workflow —
from statistical analysis and hypothesis testing to comparative modeling and final model selection.

- **Data analysis and statistics**  
  pandas, numpy — analysis of feature distributions, missing values, and the target variable;
statistical dependency testing (including eta-squared)

- **Visualization and EDA**  
  matplotlib, seaborn

- **Linear and regularized models**  
  Linear Regression (OLS), Ridge, Lasso, Elastic Net

- **Polynomial models**  
  Polynomial Regression combined with PCA and regularization

- **Distance- and margin-based methods**  
  K-Nearest Neighbors, Support Vector Machines

- **Decision trees and ensemble methods**  
  Decision Tree, Random Forest

- **Gradient boosting**  
  XGBoost, CatBoost, LightGBM

- **Experiment infrastructure**  
  Jupyter Notebook

---

## License

This project is licensed under the MIT License.