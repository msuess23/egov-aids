import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score

# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _build_clustering_preprocessor(categorical_cols: list, numeric_cols: list):
    """Builds the preprocessing pipeline dynamically based on provided columns."""
    transformers = []
    if numeric_cols:
        transformers.append(('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scal', StandardScaler())]), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols))
        
    return ColumnTransformer(transformers=transformers, remainder='drop')

def _apply_pc_protection(df: pd.DataFrame, max_rows: int):
    """Safeguards memory for O(n^2) algorithms by subsampling large datasets."""
    if len(df) > max_rows:
        print(f"  -> PC Protection: Subsampling to {max_rows:,} rows.")
        return df.sample(n=max_rows, random_state=42).copy()
    return df

# =============================================================================
# PUBLIC API
# =============================================================================

def find_optimal_k(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, max_k: int = 6):
    """Determines the best cluster count using the Silhouette Score."""
    print(f"\nSearching for optimal k (2 to {max_k})...")
    preprocessor = _build_clustering_preprocessor(categorical_cols, numeric_cols)
    
    sample_size = min(30000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    X_processed = preprocessor.fit_transform(df_sample)
    
    scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        scores.append(score)
        print(f"  -> k={k}: Silhouette = {score:.4f}")
        
    recommended_k = k_range[np.argmax(scores)]
    print(f" -> Recommended cluster count: {recommended_k}")
    return recommended_k

def train_kmeans(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, n_clusters: int = 3, n_init: int = 10, **kwargs):
    """Trains a K-Means model. Scales efficiently to millions of rows."""
    print("\nInitiating K-MEANS Clustering...")
    preprocessor = _build_clustering_preprocessor(categorical_cols, numeric_cols)
    X_processed = preprocessor.fit_transform(df)
    
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42, **kwargs)
    labels = model.fit_predict(X_processed)
    
    return df, labels, X_processed, model

def train_dbscan(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, eps: float = 0.5, min_samples: int = 5, max_rows: int = 100000, **kwargs):
    """Trains a DBSCAN model with automatic PC protection."""
    print("\nInitiating DBSCAN Clustering...")
    working_df = _apply_pc_protection(df, max_rows)
    
    preprocessor = _build_clustering_preprocessor(categorical_cols, numeric_cols)
    X_processed = preprocessor.fit_transform(working_df)
    
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, **kwargs)
    labels = model.fit_predict(X_processed)
    
    return working_df, labels, X_processed, model

def train_gmm(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, n_clusters: int = 3, covariance_type: str = 'diag', n_init: int = 1, max_rows: int = 100000, **kwargs):
    """Trains a Gaussian Mixture Model with automatic PC protection."""
    print("\nInitiating GMM Clustering...")
    working_df = _apply_pc_protection(df, max_rows)
    
    preprocessor = _build_clustering_preprocessor(categorical_cols, numeric_cols)
    X_processed = preprocessor.fit_transform(working_df)
    
    # GMM needs a dense matrix
    X_dense = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
    
    model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=42, **kwargs)
    model.fit(X_dense)
    labels = model.predict(X_dense)
    
    return working_df, labels, X_processed, model