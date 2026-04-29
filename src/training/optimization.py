import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def _build_model_pipeline(model_name: str, task: str):
    """
    Internal helper to construct the base estimator, appropriate parameter grid, 
    and preprocessing steps required by the specific mathematical nature of the model.
    """
    if model_name == 'xgboost':
        # Gradient Boosting natively handles NaNs and ignores feature scales
        estimator = xgb.XGBRegressor(random_state=42, n_jobs=-1) if task == 'reg' else xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 250, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9]
        }
        
    elif model_name == 'lightgbm':
        estimator = LGBMRegressor(random_state=42, n_jobs=-1) if task == 'reg' else LGBMClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 250, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127] # LGBM grows leaf-wise, not depth-wise
        }
        
    elif model_name == 'rf':
        # Random Forests are scale-invariant but crash on missing values (NaNs)
        base = RandomForestRegressor(random_state=42, n_jobs=-1) if task == 'reg' else RandomForestClassifier(random_state=42, n_jobs=-1)
        estimator = Pipeline([('imputer', SimpleImputer(strategy='median')), ('model', base)])
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None]
        }
        
    elif model_name == 'linear':
        # Distance/Gradient-based linear models STRICTLY require imputed and scaled data
        base = Ridge() if task == 'reg' else LogisticRegression(max_iter=1000, random_state=42)
        estimator = Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler()), 
            ('model', base)
        ])
        param_grid = {'model__alpha': [0.1, 1.0, 10.0]} if task == 'reg' else {'model__C': [0.1, 1.0, 10.0]}
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    return estimator, param_grid

def tune_supervised_model(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, model_name: str, task: str = 'reg', n_iter: int = 10):
    """
    Executes a Randomized Search Cross-Validation. Uses GroupKFold to strictly 
    prevent intra-notice data leakage during the validation folds.
    
    Args:
        task (str): 'reg' for Regression, 'clf' for Classification.
    """
    print(f"\\nInitiating Hyperparameter Tuning for '{model_name.upper()}' ({task.upper()})...")
    
    estimator, param_grid = _build_model_pipeline(model_name, task)
    
    # 3-Fold Cross Validation respecting the Notice IDs
    gkf = GroupKFold(n_splits=3)
    
    # Define optimization target based on econometric task
    scoring_metric = 'neg_root_mean_squared_error' if task == 'reg' else 'f1'
    
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring_metric,
        cv=gkf,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    print(f" -> Best parameters identified: {search.best_params_}")
    
    return search.best_estimator_

def find_optimal_clusters(X_scaled: pd.DataFrame, max_k: int = 8) -> KMeans:
    """
    Iteratively tests different K values for K-Means clustering to find the 
    mathematical 'sweet spot' (highest Silhouette Score) representing market archetypes.
    """
    print(f"\\nSearching for optimal market clusters (K=2 to {max_k})...")
    
    best_k = 2
    best_score = -1
    best_model = None
    
    # We take a random sample to calculate silhouette score (it scales quadratically, too slow on 2M rows)
    sample_size = min(50000, len(X_scaled))
    X_sample = X_scaled.sample(n=sample_size, random_state=42)
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate score on the sample for performance reasons
        sample_labels = kmeans.predict(X_sample)
        score = silhouette_score(X_sample, sample_labels)
        
        print(f" -> K={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            
    print(f"\\nOptimal Market Archetypes found: K={best_k} (Score: {best_score:.4f})")
    return best_model