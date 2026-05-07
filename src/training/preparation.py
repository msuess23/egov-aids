import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def perform_group_split(df: pd.DataFrame, target_col: str, group_col: str, drop_cols: list, test_size: float = 0.2) -> tuple:
    """
    Partitions the dataset into training and testing sets while preserving 
    procurement structures (all lots of a notice stay in the same set).
    This strictly prevents data leakage across correlated observations.
    """
    print(f"Executing Group-based Split (Target: '{target_col}', Grouping by: '{group_col}')...")
    
    # Initialize the group splitter to ensure notice-integrity
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    y_train = train_data[target_col]
    y_test = test_data[target_col]
    
    # Extract group series for Train to enable safe Cross-Validation later
    groups_train = train_data[group_col]
    
    # Isolate features by dropping targets, groups, and explicit leakage columns
    exclusion_list = drop_cols + [target_col, group_col, 'ID_LOT']
    X_train = train_data.drop(columns=[c for c in exclusion_list if c in train_data.columns])
    X_test = test_data.drop(columns=[c for c in exclusion_list if c in test_data.columns])
    
    print(f" -> Split successful. Train: {len(X_train):,} rows | Test: {len(X_test):,} rows.")
    return X_train, X_test, y_train, y_test, groups_train


def build_preprocessor(categorical_cols: list, numeric_cols: list):
    """
    Global, dynamic preprocessing pipeline used across all ML tasks
    (Clustering, Classification, Regression).
    
    Domain-specific imputation (like 'MISSING' or 0) is assumed to be handled 
    upstream in Notebook 05. This serves as the mathematical transformation engine 
    and safety net for any unexpected NaNs.
    """
    transformers = []
    
    if numeric_cols:
        transformers.append(('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')), 
            ('log', FunctionTransformer(np.log1p)), 
            ('scal', StandardScaler())
        ]), numeric_cols))
        
    if categorical_cols:
        transformers.append(('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ]), categorical_cols))
        
    return ColumnTransformer(transformers=transformers, remainder='drop')