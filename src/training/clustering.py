import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from src.training.preparation import build_preprocessor

# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================

def _apply_pc_protection(df: pd.DataFrame, max_rows: int):
    """
    Restricts the dataset size for algorithms with quadratic space/time complexity.
    Computing pairwise distance matrices for density-based or probabilistic models 
    exceeds standard memory limits when applied to millions of procurement records.
    """
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42).copy()
    return df

# =============================================================================
# PUBLIC API
# =============================================================================

def find_optimal_k(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, max_k: int = 6):
    """
    Evaluates the optimal number of segments (k) using the Silhouette Score.
    This metric identifies the configuration where procurement events are highly 
    cohesive within their segment while being clearly separated from other segments.
    A representative sample is used to avoid computational bottlenecks.
    """
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    X_processed = preprocessor.fit_transform(df_sample)
    
    scores = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_processed)
        scores.append(silhouette_score(X_processed, labels))
        
    return k_range[np.argmax(scores)]


def train_kmeans(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, n_clusters: int = 3, n_init: int = 15, **kwargs):
    """
    Partitions the dataset into spherical segments based on centroid proximity.
    This provides a highly scalable baseline for market segmentation, capable of 
    processing the entire procurement dataset efficiently.
    """
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    X_processed = preprocessor.fit_transform(df)
    
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42, **kwargs)
    labels = model.fit_predict(X_processed)
    
    return df, labels, X_processed, model


def train_dbscan(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, eps: float = 0.5, min_samples: int = 5, max_rows: int = 100000, **kwargs):
    """
    Isolates high-density core markets from individual, customized outliers.
    Procurement data contains many highly specific mega-projects. Density-based 
    clustering categorizes these outliers as noise (-1) rather than forcing them 
    into standard market segments.
    """
    working_df = _apply_pc_protection(df, max_rows)
    
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    X_processed = preprocessor.fit_transform(working_df)
    
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, **kwargs)
    labels = model.fit_predict(X_processed)
    
    return working_df, labels, X_processed, model


def train_gmm(df: pd.DataFrame, categorical_cols: list, numeric_cols: list, n_clusters: int = 3, covariance_type: str = 'diag', n_init: int = 1, max_rows: int = 100000, **kwargs):
    """
    Models procurement segments using probabilistic distributions.
    Financial data often shows correlations (e.g., volume increasing with duration).
    Gaussian Mixture Models with full covariance matrices can capture these elliptical 
    data shapes, which distance-based models might split incorrectly.
    """
    working_df = _apply_pc_protection(df, max_rows)
    
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    X_processed = preprocessor.fit_transform(working_df)
    
    X_dense = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
    
    model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, random_state=42, **kwargs)
    model.fit(X_dense)
    labels = model.predict(X_dense)
    
    return working_df, labels, X_processed, model