import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)

def evaluate_regression(y_true, y_pred, n_features, is_log_transformed=True):
    """
    Calculates regression metrics with a focus on robustness.
    
    Domain Knowledge:
    - Median Absolute Error (MedAE) is preferred over MAE as contract prices 
      in TED data feature extreme outliers that would skew standard averages.
    """
    y_t = np.expm1(y_true) if is_log_transformed else y_true
    y_p = np.expm1(y_pred) if is_log_transformed else y_pred

    mae = mean_absolute_error(y_t, y_p)
    medae = median_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    
    # Adjust R-squared for feature count to identify potential over-specification
    adj_r2 = 1 - (1 - r2) * (len(y_t) - 1) / (len(y_t) - n_features - 1)
    
    print("\n--- Regression Results ---")
    print(f"MedAE (Robust): €{medae:,.2f}")
    print(f"MAE:            €{mae:,.2f}")
    print(f"RMSE:           €{rmse:,.2f}")
    print(f"Adj. R-squared: {adj_r2:.4f}")
    
    return {'medae': medae, 'adj_r2': adj_r2}

def evaluate_classification(y_true, y_pred):
    """
    Calculates classification metrics for binary tasks.
    
    Domain Knowledge:
    - Balanced Accuracy is critical as 'Market Failure' classes are often 
      unbalanced (fewer competitive tenders than single bids).
    """
    results = {
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }
    
    print("\n--- Classification Results ---")
    print(f"Balanced Acc: {results['bal_acc']:.4f}")
    print(f"F1-Score:     {results['f1']:.4f}")
    print(f"Precision:    {results['precision']:.4f}")
    print(f"Recall:       {results['recall']:.4f}")
    
    return results

def evaluate_clustering(X_processed, labels, sample_size=20000):
    """
    Evaluates cluster density and separation using multiple scientific metrics.
    Calculations are sampled to ensure fast execution on large datasets.
    """
    valid_mask = labels != -1
    X_valid = X_processed[valid_mask]
    labels_valid = labels[valid_mask]
    
    if len(set(labels_valid)) < 2:
        return {'silhouette': -1, 'davies_bouldin': -1, 'calinski': -1}
    
    n_total = X_valid.shape[0]
    idx = np.random.choice(n_total, min(sample_size, n_total), replace=False)
    
    X_sample = X_valid[idx]
    y_sample = labels_valid[idx]
    
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()
    
    # Metriken berechnen
    sil_score = silhouette_score(X_sample, y_sample, random_state=42)
    db_score = davies_bouldin_score(X_sample, y_sample)
    ch_score = calinski_harabasz_score(X_sample, y_sample)
    
    print("\n--- Clustering Metrics ---")
    print(f"Silhouette Score:      {sil_score:.4f} (Higher is better, max 1)")
    print(f"Davies-Bouldin Index:  {db_score:.4f} (Lower is better, min 0)")
    print(f"Calinski-Harabasz:     {ch_score:,.1f} (Higher is better)")
    
    return {'silhouette': sil_score, 'davies_bouldin': db_score, 'calinski': ch_score}