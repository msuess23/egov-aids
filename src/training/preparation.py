import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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

def prepare_clustering_data(df: pd.DataFrame, features: list) -> tuple:
    """
    Prepares data specifically for distance-based Unsupervised Learning (e.g., K-Means).
    Since distance algorithms fail on NaNs and are biased by different scales 
    (e.g., Days vs. Euros), we must impute and standardize.
    """
    print("Preparing features for Unsupervised Clustering (Imputation & Scaling)...")
    valid_cols = [c for c in features if c in df.columns]
    X_raw = df[valid_cols].copy()
    
    # Build a rigid preprocessing pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_scaled = pipeline.fit_transform(X_raw)
    
    print(f" -> Scaled {len(valid_cols)} features for {len(X_scaled):,} observations.")
    return pd.DataFrame(X_scaled, columns=valid_cols, index=df.index), pipeline