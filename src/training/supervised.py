import pandas as pd
import numpy as np
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Ridge, LogisticRegression, Lasso
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def _build_model_pipeline(model_name: str, task: str, categorical_cols: list, numeric_cols: list):
    """
    Constructs a scikit-learn pipeline tailored to the mathematical requirements 
    of the specific model family. 
    
    Domain Knowledge: 
    - Tree-based models (Family A) use Ordinal Encoding as they split based on 
      thresholds rather than calculating distances.
    - Linear/Distance models (Family B) require One-Hot Encoding and Scaling 
      to ensure that feature magnitudes or arbitrary category IDs do not 
      distort the gradient descent or distance metrics.
    """
    
    # Basic imputation for numerical features to handle missing procurement data
    numeric_transformer_simple = SimpleImputer(strategy='median')
    
    # Scaling is mandatory for distance-sensitive models (Family B)
    numeric_transformer_scaled = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # FAMILY A: Tree-based (Invariant to scale, handles ordinal relations)
    preprocessor_trees = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_simple, numeric_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ],
        remainder='drop'
    )

    # FAMILY B: Linear / Distance (Sensitive to scale and nominal category IDs)
    preprocessor_linear = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_scaled, numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ],
        remainder='drop'
    )

    # Model and Parameter Grid Mapping
    if model_name == 'xgboost':
        base = xgb.XGBRegressor(random_state=42, n_jobs=-1) if task == 'reg' else xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        param_grid = {'model__n_estimators': [100, 250, 500], 'model__learning_rate': [0.01, 0.05, 0.1], 'model__max_depth': [5, 7, 9]}
        
    elif model_name == 'rf':
        base = RandomForestRegressor(random_state=42, n_jobs=-1) if task == 'reg' else RandomForestClassifier(random_state=42, n_jobs=-1)
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None]}
        
    elif model_name == 'dt':
        base = DecisionTreeRegressor(random_state=42) if task == 'reg' else DecisionTreeClassifier(random_state=42)
        estimator = Pipeline([('preprocessor', preprocessor_trees), ('model', base)])
        param_grid = {'model__max_depth': [5, 10, 15], 'model__min_samples_leaf': [1, 5, 20]}
        
    elif model_name == 'linear':
        base = Ridge() if task == 'reg' else LogisticRegression(max_iter=1000, random_state=42)
        estimator = Pipeline([('preprocessor', preprocessor_linear), ('model', base)])
        param_grid = {'model__alpha': [0.1, 1.0, 10.0]} if task == 'reg' else {'model__C': [0.1, 1.0, 10.0]}
        
    elif model_name == 'lasso':
        base = Lasso(random_state=42, max_iter=2000)
        estimator = Pipeline([('preprocessor', preprocessor_linear), ('model', base)])
        param_grid = {'model__alpha': [0.01, 0.1, 1.0]}
        
    return estimator, param_grid

def prepare_supervised_data(df: pd.DataFrame, target_col: str, group_col: str, feature_cols: list, test_size: float = 0.2):
    """
    Splits data safely using GroupShuffleSplit to prevent data leakage 
    across the same notice (Notice-Lot hierarchy).
    """
    print(f"\nPreparing data for supervised learning. Target: '{target_col}'")
    
    # Drop rows where target is missing
    df_clean = df.dropna(subset=[target_col]).copy()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    groups = df_clean[group_col]
    
    # Splitting based on groups (Notices)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    
    print(f" -> Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, groups_train

def tune_supervised_model(X_train, y_train, groups_train, categorical_cols, numeric_cols, model_name, task='reg', n_iter=10):
    """
    Executes Randomized Search CV with GroupKFold validation.
    
    Domain Knowledge: 
    - GroupKFold is used to respect the hierarchical nature of TED data (Notices vs. Lots).
    - It ensures that lots belonging to the same notice do not cross-pollinate 
      between training and validation folds, preventing unrealistic performance metrics.
    """
    print(f"\nTuning Hyperparameters for '{model_name.upper()}' ({task.upper()})...")
    
    estimator, param_grid = _build_model_pipeline(model_name, task, categorical_cols, numeric_cols)
    group_kfold = GroupKFold(n_splits=3)
    scoring_metric = 'neg_root_mean_squared_error' if task == 'reg' else 'f1'
    
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring_metric,
        cv=group_kfold,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    print(f" -> Optimization complete. Best parameters: {search.best_params_}")
    
    return search.best_estimator_