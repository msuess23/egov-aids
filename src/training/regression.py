import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.training.preparation import build_preprocessor

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_regression_data(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    """
    All-in-One Data Preparation for Cost Overrun Analysis.
    
    Methodological steps:
    1. Notice-Level Aggregation: Compares holistic project budgets.
    2. Fair-Subset Testing & Zero-Value Protection: Drops records without estimation 
       AND mathematically impossible/corrupted zero values to prevent log(0) = -inf.
    3. Target Calculation: Computes the deviation ratio.
    4. Log-Transformation: Applied to the ratio for symmetric error weighting.
    """
    print("Preparing data for Budget Deviation Analysis...")
    
    # 1. Reduce to Notice-level
    df_notice = df.drop_duplicates(subset=['ID_NOTICE']).copy()
    
    # 2. Fair-Subset Testing & Mathematical Protection
    # We strictly require both values to be > 0.
    # Estimated > 0 prevents "Division by Zero"
    # Award > 0 prevents the ratio from being 0 (which would cause np.log(0) = -inf)
    df_reg = df_notice[(df_notice['TARGET_AWARD_VALUE_EUR'] > 0) & 
                       (df_notice['ESTIMATED_VALUE_EUR'] > 0)].copy()
    
    # 3. Calculate deviation ratio (1.0 = perfect estimation)
    df_reg['TARGET_BUDGET_DEVIATION_RATIO'] = df_reg['TARGET_AWARD_VALUE_EUR'] / df_reg['ESTIMATED_VALUE_EUR']
    
    print(f"Extraction complete. Notices viable for Budget Analysis: {len(df_reg):,}")
    
    # 4. Feature Selection and Target Transformation
    X = df_reg[feature_cols]
    y = np.log(df_reg['TARGET_BUDGET_DEVIATION_RATIO'])
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# MODELING PIPELINES
# =============================================================================

def tune_regression_model(X_train: pd.DataFrame, y_train: pd.Series, 
                          categorical_cols: list, numeric_cols: list, model_name: str):
    """
    Executes GridSearchCV to find optimal hyperparameters for the selected algorithm.
    Supported models: 'lasso' (Linear), 'rf' (Bagging), 'xgboost' (Boosting).
    """
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    
    models = {
        'lasso': {
            'estimator': Lasso(random_state=42, max_iter=2000),
            'params': {'regressor__alpha': [0.001, 0.01, 0.1, 1.0]}
        },
        'rf': {
            'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [10, 20]
            }
        },
        'xgboost': {
            'estimator': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'params': {
                'regressor__n_estimators': [100, 300],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__max_depth': [5, 7],
                'regressor__subsample': [0.8, 1.0]
            }
        }
    }
    
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not supported.")
        
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', models[model_name]['estimator'])
    ])
    
    print(f"Starting GridSearch for {model_name.upper()}...")
    search = GridSearchCV(
        pipeline, 
        param_grid=models[model_name]['params'], 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    print(f"Best params for {model_name.upper()}: {search.best_params_}")
    return search.best_estimator_