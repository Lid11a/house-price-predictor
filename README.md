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

## Project Structure
```
house-price-predictor/
│
├── data/                 # Исходные данные (train/test)
    ├── train/            # Тренировочные данные
    ├── train/            # Тренировочные данные
    ├── train/            # Тренировочные данные
├── images/               # Сохранённые картинки из ноутбука (графики, визуализации)
├── notebooks/            # Jupyter notebook со всем кодом
└── submission/           # Финальный файл с предсказаниями для Kaggle
    └── submission.csv
├── requirements.txt      # Список зависимостей проекта
├── README.md             # Этот файл с описанием проекта
├── .gitignore            # Файл для исключения ненужных файлов из Git
└── LICENSE               # Лицензия проекта
```
