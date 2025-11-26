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
To reproduce the analysis and modeling results, you need Python 3.9+ and the project dependencies.
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Lid11a/house-price-predictor
    cd house-price-predictor
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS / Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Data preparation:**
    The dataset is already included in the ./data/ directory. No additional downloads are required.

5. **Launch the notebook**
     Start Jupyter Lab or Jupyter Notebook from the project root directory:
  ```bash
  jupyter notebook
  # or
  jupyter lab
  ```
6. **Run the notebook**
   Open `notebooks/house-price-predictor.ipynb` and execute all cells sequentially.

7. **Generate submission file**
   The final cell blocks will automatically select the best model and generate the final predictions file. The output file is saved in the `./submission/` directory as a CSV file. The file name reflects the best-performing model.

## Results (Результаты): Ваши лучшие метрики.

## License (Лицензия).
