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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

def _build_model_pipeline(model_name: str, task: str, categorical_cols: list, numeric_cols: list):
    """
    Internal helper to construct the base estimator, appropriate parameter grid, 
    and preprocessing steps required by the specific mathematical nature of the model.
    """
    # -------------------------------------------------------------------------
    # 1. Define feature-specific transformers
    # -------------------------------------------------------------------------
    # Numeric features always get imputed (Trees don't strictly need it if XGBoost, 
    # but Random Forest and linear models do).
    numeric_transformer = SimpleImputer(strategy='median')
    
    # Numeric transformer with scaling for distance/gradient-based models
    numeric_transformer_scaled = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # -------------------------------------------------------------------------
    # 2. Build Family-Specific Preprocessors
    # -------------------------------------------------------------------------
    
    # FAMILY A: Tree-based models (Invariant to scale, need Ordinal Encoding)
    preprocessor_trees = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            # OrdinalEncoder converts strings to integers. handle_unknown='use_encoded_value' 
            # prevents crashes if new categories appear in the test set.
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ],
        remainder='drop'
    )

    # FAMILY B: Linear / Distance models (Sensitive to scale, need One-Hot Encoding)
    preprocessor_linear = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_scaled, numeric_cols),
            # OneHotEncoder creates binary dummy variables. drop='first' avoids multicollinearity (dummy variable trap).
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ],
        remainder='drop'
    )

    # -------------------------------------------------------------------------
    # 3. Assemble Final Pipelines & Param Grids
    # -------------------------------------------------------------------------
    if model_name == 'xgboost':
        base = xgb.XGBRegressor(random_state=42, n_jobs=-1) if task == 'reg' else xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        param_grid = {
            'model__n_estimators': [100, 250, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [5, 7, 9]
        }
        
    elif model_name == 'lightgbm':
        base = LGBMRegressor(random_state=42, n_jobs=-1) if task == 'reg' else LGBMClassifier(random_state=42, n_jobs=-1)
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        param_grid = {
            'model__n_estimators': [100, 250, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__num_leaves': [31, 63, 127]
        }
        
    elif model_name == 'rf':
        base = RandomForestRegressor(random_state=42, n_jobs=-1) if task == 'reg' else RandomForestClassifier(random_state=42, n_jobs=-1)
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None]
        }
        
    elif model_name == 'linear':
        base = Ridge() if task == 'reg' else LogisticRegression(max_iter=1000, random_state=42)
        estimator = Pipeline([('preprocessor', preprocessor_linear), ('model', base)])
        param_grid = {'model__alpha': [0.1, 1.0, 10.0]} if task == 'reg' else {'model__C': [0.1, 1.0, 10.0]}

    elif model_name == 'lasso': # Für Regression
        base = Lasso(random_state=42, max_iter=2000)
        estimator = Pipeline([('preprocessor', preprocessor_linear), ('model', base)])
        param_grid = {'model__alpha': [0.01, 0.1, 1.0, 10.0]}
        
    elif model_name == 'dt': # Decision Tree
        base = DecisionTreeRegressor(random_state=42) if task == 'reg' else DecisionTreeClassifier(random_state=42)
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        # Pruning parameters to prevent overfitting (Vorlesungsskript 4.3.3.3)
        param_grid = {
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 10, 50],
            'model__min_samples_leaf': [1, 5, 20]
        }
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    return estimator, param_grid

def train_clustering_model(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, model_name: str, subsample: bool = False, **kwargs):
    """
    Preprocesses and trains a clustering algorithm. 
    Automatically handles subsampling for heavy algorithms (DBSCAN/GMM) 
    if requested or if data is too large to prevent MemoryErrors.
    """
    print(f"\nInitiating Clustering for '{model_name.upper()}'...")
    
    # -------------------------------------------------------------------------
    # 1. PC Protection Logic (Auto-Subsampling for O(n^2) or dense matrix algorithms)
    # -------------------------------------------------------------------------
    working_df = df
    if subsample or (model_name in ['dbscan', 'gmm'] and len(df) > 100000):
        sample_n = min(100000, len(df))
        print(f"  -> PC Protection: Subsampling to {sample_n:,} rows for {model_name.upper()}.")
        working_df = df.sample(n=sample_n, random_state=42).copy()

    # -------------------------------------------------------------------------
    # 2. Preprocessing for Distance/Variance-Based Models (Family B)
    # -------------------------------------------------------------------------
    # Numeric features need scaling, categorical features need One-Hot-Encoding
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')), 
            ('scal', StandardScaler())
        ]), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])
    
    print("  -> Preprocessing data (One-Hot-Encoding & Scaling)...")
    X_processed = preprocessor.fit_transform(working_df)
    
    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    if model_name == 'kmeans':
        k = kwargs.get('n_clusters', 3)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_processed)
        
    elif model_name == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X_processed)
        
    elif model_name == 'gmm':
        k = kwargs.get('n_clusters', 3)
        model = GaussianMixture(n_components=k, random_state=42, covariance_type='diag')
        
        # FIX: GMM requires dense data instead of a sparse matrix (from OneHotEncoder).
        # Since we subsampled, converting to array will not crash the RAM.
        print("  -> Converting sparse matrix to dense array for GMM...")
        X_dense = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
        
        model.fit(X_dense)
        labels = model.predict(X_dense)
        
    else:
        raise ValueError(f"Unknown clustering model: {model_name}")
        
    # -------------------------------------------------------------------------
    # 4. Basic Console Reporting
    # -------------------------------------------------------------------------
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  -> Algorithm finished. Found {n_clusters_found} clusters.")
    
    if -1 in labels:
        noise_points = list(labels).count(-1)
        print(f"  -> Note: DBSCAN classified {noise_points:,} points as noise (-1).")
        
    return working_df, labels, X_processed, model

def tune_supervised_model(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, categorical_cols: list, numeric_cols: list, model_name: str, task: str = 'reg', n_iter: int = 10):
    """
    Executes a Randomized Search Cross-Validation. Uses GroupKFold to strictly 
    prevent intra-notice data leakage during the validation folds.
    
    Args:
        task (str): 'reg' for Regression, 'clf' for Classification.
    """
    print(f"\\nInitiating Hyperparameter Tuning for '{model_name.upper()}' ({task.upper()})...")
    
    estimator, param_grid = _build_model_pipeline(model_name, task, categorical_cols, numeric_cols)
    
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

def find_optimal_k(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, max_k: int = 8):
    """
    Iterates through k values to find the optimal cluster count using Silhouette Score.
    Uses a sample of the data to keep it performant.
    """
    print(f"\nSearching for optimal k (1 to {max_k})...")
    
    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scal', StandardScaler())]), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])
    
    # We sample for the search to save time
    df_sample = df.sample(n=min(50000, len(df)), random_state=42)
    X_proc = preprocessor.fit_transform(df_sample)
    
    scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X_proc)
        score = silhouette_score(X_proc, labels)
        scores.append(score)
        print(f"  -> k={k}: Silhouette Score = {score:.4f}")
        
    best_k = k_range[np.argmax(scores)]
    print(f" -> Recommended k based on Silhouette Score: {best_k}")
    return best_k

def train_clustering_model(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, model_name: str, subsample: bool = False, **kwargs):
    """
    Preprocesses and trains clustering. Automatically handles subsampling for 
    heavy algorithms (DBSCAN/GMM) if requested or if data is too large.
    """
    print(f"\nInitiating Clustering for '{model_name.upper()}'...")
    
    # Logic for PC safety: auto-subsample for complex models if data is huge
    working_df = df
    if subsample or (model_name in ['dbscan', 'gmm'] and len(df) > 100000):
        sample_n = 100000
        print(f"  -> PC Protection: Subsampling to {sample_n:,} rows for {model_name.upper()}.")
        working_df = df.sample(n=sample_n, random_state=42).copy()

    # Preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scal', StandardScaler())]), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])
    
    X_processed = preprocessor.fit_transform(working_df)
    
    if model_name == 'kmeans':
        model = KMeans(n_clusters=kwargs.get('n_clusters', 3), random_state=42, n_init=10)
        labels = model.fit_predict(X_processed)
    elif model_name == 'dbscan':
        model = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5), n_jobs=-1)
        labels = model.fit_predict(X_processed)
    elif model_name == 'gmm':
        model = GaussianMixture(n_components=kwargs.get('n_clusters', 3), random_state=42, covariance_type='diag')
        model.fit(X_processed)
        labels = model.predict(X_processed)
        
    return working_df, labels, X_processed, model