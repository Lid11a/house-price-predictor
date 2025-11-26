# Advanced regression for house price prediction  
A high-performance machine learning framework to predict house prices using the Kaggle dataset. Goes beyond standard regression by combining multiple models, smart pipelines, and rigorous evaluation.

## Project overview  
The core architecture and modeling approach rely on these key features:
1. Smart Preprocessing
- Handles missing data, engineers features, and encodes categorical variables safely using Pipeline and ColumnTransformer.
- Prevents data leakage and ensures consistent transformations across all models.
2. Flexible Pipelines
- Supports Standard, PCA, No-Collinearity, and Polynomial Features pipelines.
- Runs each model in an optimized configuration for its strengths.
- Makes experimenting with different feature sets easy.
3. Hyperparameter Optimization
- Uses GridSearchCV for fine-tuning and RandomizedSearchCV for broad exploration.
- Ensures models achieve optimal performance efficiently.
4. Feature Insights
- Applies Permutation Importance (PFI) to rank features across all models, including complex ones like SVR and K-NN.
- Provides interpretable results even for high-performance gradient boosting models.
5. RMSLE-Optimized Performance
- Evaluates all pipelines using Root Mean Squared Logarithmic Error (RMSLE).
- Guarantees accurate predictions and strong generalization on unseen data.

## Project structure
```
house-price-predictor/
│
├── data/                                # Original dataset files
│    ├── data_description.txt            # Description of dataset columns and features
│    ├── sample_submission.csv           # Sample submission file from Kaggle
│    ├── test.csv                        # Test set
│    └── train.csv                       # Training set
│
├── images/                              # Saved plots and visualizations from notebook
│    ├── 1.sale_price_comparison.png     
│    ├── 2.full_collinearity_heatmap.png  
│    ├── 3.rmsle_comparison.png   
│    ├── 4.actual_vs_predicted.png   
│    ├── 5.residual_plot_1.png 
│    ├── 6.residual_plot_2.png   
│    ├── 7.residual_plot_3.png  
│    └── 8.feature_importance_universal.png
│
├── notebooks/                           # Jupyter notebook containing full project code
│    └── house-price-predictor.ipynb
│
└── submission/                          # Final predictions for Kaggle
│    └── submission_CatBoost.csv
│
├── requirements.txt                     # Project dependencies
├── README.md                            # Project overview, instructions, and details
├── .gitignore                           # Files and folders excluded from Git
└── LICENSE                              # Project license
```

## Installation and usage
To reproduce the analysis and modeling results, you need Python 3.9 - 3.11 and the project dependencies.
1.  **Clone the repository**
    ```bash
    git clone https://github.com/Lid11a/house-price-predictor
    cd house-price-predictor
    ```
2.  **Create and activate a virtual environment**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS / Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data preparation**  
    The dataset is already included in the ./data/ directory. No additional downloads are required.

5.  **Launch the notebook**  
    Start Jupyter Lab or Jupyter Notebook from the project root directory:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
7. **Run the notebook**  
   Open `notebooks/house-price-predictor.ipynb` and execute all cells sequentially.

8. **Generate submission file**  
   The final cell blocks will automatically select the best model and generate the final predictions file. The output file is saved in the `./submission/` directory as a CSV file. The file name reflects the best-performing model.

## Results (Результаты): Ваши лучшие метрики  
The objective of this project was to predict house prices using advanced ensemble regression techniques and comprehensive feature engineering. The final pipeline included a comparative analysis of **21 different models**, with **CatBoostRegressor** demonstrating the best performance on the test set.  
The following table summarizes the performance metrics of all 21 regression models evaluated, sorted by the target metric, RMSLE.  
```
| Model | RMSLE | RMSE ($) | R² | MAE ($) |
| :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | **0.125271** | **24,822** | **0.912050** | 14,760 |
| LightGBM | 0.128015 | 25,633 | 0.906208 | 15,213 |
| SVR\_PCA | 0.128608 | 28,088 | 0.887380 | 14,633 |
| SVR | 0.128766 | 27,882 | 0.889024 | 14,509 |
| XGBoost | 0.129106 | 25,123 | 0.909905 | 15,206 |
| Lasso | 0.129990 | 23,925 | 0.918293 | 15,664 |
| ElasticNet | 0.129990 | 23,925 | 0.918293 | 15,664 |
| SVR\_NC | 0.131168 | 27,693 | 0.890528 | 15,291 |
| Ridge | 0.133576 | 25,086 | 0.910169 | 16,299 |
| OLS\_PCA | 0.136912 | 26,021 | 0.903347 | 17,402 |
| RandomForest | 0.138850 | 27,455 | 0.892397 | 16,579 |
| PolyRidge\_PCA | 0.143246 | 29,941 | 0.872032 | 16,987 |
| PolyElasticNet\_PCA | 0.143345 | 29,321 | 0.877276 | 16,497 |
| PolyLasso\_PCA | 0.143907 | 29,308 | 0.877385 | 16,489 |
| DecisionTree | 0.145511 | 30,218 | 0.869652 | 17,513 |
| Poly\_OLS\_PCA | 0.171691 | 34,009 | 0.834900 | 19,441 |
| K-NN\_PCA | 0.172111 | 37,771 | 0.796343 | 20,953 |
| K-NN\_NC | 0.178595 | 36,335 | 0.811542 | 21,095 |
| KNN | 0.179674 | 35,325 | 0.821874 | 20,945 |
| PolyOLS\_NC | 0.200042 | 94,041 | -0.262430 | 22,405 |
| OLS\_NC | 0.202470 | 35,505 | 0.820047 | 23,886 |
```
The final deployed CatBoost model achieved an internal RMSLE of 0.125271, with a Kaggle Public Score of 0.12694 on the submission platform, demonstrating strong predictive accuracy and feature robustness.

## License (Лицензия)  
This project is licensed under the **MIT License**.
