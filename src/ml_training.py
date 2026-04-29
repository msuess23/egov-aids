# ==============================================================================
# src/ml_training.py (Machine Learning Architecture)
# ==============================================================================
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt

def perform_group_split(df: pd.DataFrame, target_col: str, group_col: str, drop_cols: list, test_size: float = 0.2) -> tuple:
    """
    Partitions the dataset into training and testing sets while preserving 
    group structures (e.g., all lots of a notice stay in the same set).
    Prevents data leakage across correlated observations.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The column to be predicted.
        group_col (str): The column used to group data (e.g., Notice ID).
        drop_cols (list): Columns to exclude from the feature matrix (e.g., alternative targets, IDs).
        test_size (float): The proportion of the dataset to include in the test split.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Executing Group-based Split (Target: '{target_col}', Grouping by: '{group_col}')...")
    
    # Initialize the group splitter
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    
    # Generate split indices based on the grouping variable
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    y_train = train_data[target_col]
    y_test = test_data[target_col]

    groups_train = train_data[group_col]
    
    # Exclude target, group identifiers, and specified leakage columns from features
    exclusion_list = drop_cols + [target_col, group_col, 'ID_LOT']
    X_train = train_data.drop(columns=[c for c in exclusion_list if c in train_data.columns])
    X_test = test_data.drop(columns=[c for c in exclusion_list if c in test_data.columns])
    
    print(f" -> Split successful. Train Set: {len(X_train):,} rows | Test Set: {len(X_test):,} rows.")
    
    return X_train, X_test, y_train, y_test, groups_train

def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict = None) -> xgb.XGBRegressor:
    """
    Initializes and trains a Gradient Boosting Regressor (XGBoost) optimized 
    for tabular data with native missing value handling.
    """
    print("Training XGBoost Regressor...")
    
    if model_params is None:
        model_params = {
            'n_estimators': 250,          # Number of boosting rounds
            'learning_rate': 0.05,        # Step size shrinkage
            'max_depth': 7,               # Maximum tree depth
            'subsample': 0.8,             # Subsample ratio of the training instances
            'colsample_bytree': 0.8,      # Subsample ratio of columns when constructing each tree
            'random_state': 42,
            'n_jobs': -1                  # Utilize all CPU cores
        }
        
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    
    print(" -> Model training complete.")
    return model

def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """
    Predicts outcomes on the test set and calculates performance metrics.
    Automatically reverses the log1p transformation to report metrics in their 
    original economic scale (e.g., Euros, Count).
    """
    print("Evaluating Model Performance...")
    
    # Predict in log-scale
    preds_log = model.predict(X_test)
    r2 = r2_score(y_test, preds_log)
    
    # Reverse log1p transformation for interpretable business metrics
    y_test_orig = np.expm1(y_test)
    preds_orig = np.expm1(preds_log)
    
    # Ensure physical limits (no negative predictions)
    preds_orig = np.maximum(0, preds_orig) 
    
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae_orig = mean_absolute_error(y_test_orig, preds_orig)
    
    print(f" -> R² Score (Log-Scale): {r2:.4f} (Explains {r2*100:.1f}% of variance)")
    print(f" -> RMSE (Original Scale): {rmse_orig:,.2f}")
    print(f" -> MAE  (Original Scale): {mae_orig:,.2f}")
    
    return preds_orig

def plot_feature_importance(model: xgb.XGBRegressor, X_train: pd.DataFrame, top_n: int = 15, title: str = "Feature Importance"):
    """
    Visualizes the most influential features learned by the XGBoost algorithm.
    """
    importance = model.feature_importances_
    features = X_train.columns
    
    df_imp = pd.DataFrame({'Feature': features, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='#1f77b4')
    plt.title(f"{title} (Top {top_n})")
    plt.xlabel("Relative Importance (Gain)")
    plt.tight_layout()
    plt.show()

def train_lightgbm_model(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
    """
    Trains a LightGBM Regressor. Highly efficient tree-based model that 
    competes directly with XGBoost and handles NaNs natively.
    """
    print("Training LightGBM Regressor...")
    model = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(" -> LightGBM training complete.")
    return model

def train_random_forest_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains a Random Forest Regressor using a Pipeline to handle missing values 
    (Imputation), since sklearn's RF cannot handle NaNs natively.
    """
    print("Training Random Forest Pipeline (includes Median Imputation)...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestRegressor(
            n_estimators=100, # Weniger Bäume, da RF parallel und RAM-intensiv ist
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    print(" -> Random Forest training complete.")
    return pipeline

def train_ridge_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Trains a Ridge Regression (Linear Model) as a baseline.
    Requires both imputation for NaNs and Standard Scaling for coefficients.
    """
    print("Training Ridge Regression Pipeline (includes Imputation & Scaling)...")
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    pipeline.fit(X_train, y_train)
    print(" -> Ridge Regression training complete.")
    return pipeline

def tune_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, n_iter: int = 20) -> xgb.XGBRegressor:
    """
    Performs a Randomized Search Cross-Validation to find the best hyperparameters.
    Uses GroupKFold to strictly prevent data leakage between lots of the same notice.
    """
    print(f"\nStarting Systematic Hyperparameter Tuning ({n_iter} combinations)...")
    
    # 1. Definiere den Suchraum (Welche Stellschrauben wollen wir drehen?)
    param_dist = {
        'n_estimators': [100, 250, 500, 750],           # Wie viele Bäume?
        'learning_rate': [0.01, 0.05, 0.1, 0.2],        # Wie schnell lernt er?
        'max_depth': [5, 7, 9, 12],                     # Wie tief sind die Bäume?
        'subsample': [0.6, 0.8, 1.0],                   # % der Zeilen pro Baum
        'colsample_bytree': [0.6, 0.8, 1.0]             # % der Spalten pro Baum
    }
    
    # 2. Setup Group-Based Cross-Validation (Der Schutz vor Schummeln!)
    gkf = GroupKFold(n_splits=3)
    
    # 3. Das Basis-Modell
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # 4. Der Such-Algorithmus
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_root_mean_squared_error', # Wir suchen den geringsten RMSE
        cv=gkf,                                # Nutze den Gruppen-Splitter
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Starte die Suche (Hier übergeben wir zwingend die groups!)
    random_search.fit(X_train, y_train, groups=groups_train)
    
    print("\n -> TUNING COMPLETE!")
    print(f" -> Best Parameters found: {random_search.best_params_}")
    
    # Gib das fertig trainierte, beste Modell zurück
    return random_search.best_estimator_