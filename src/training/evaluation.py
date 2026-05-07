import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, median_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score, classification_report, roc_auc_score, 
    average_precision_score
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

def evaluate_classification(y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None):
    """
    Quantifies model performance with a strict focus on class imbalance metrics.
    Global accuracy is highly misleading in imbalanced procurement datasets. 
    The evaluation emphasizes Precision, Recall, and the Precision-Recall AUC (PR-AUC) 
    to accurately measure the model's ability to detect the minority class (Single Bidding).
    """
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    metrics = {}
    if y_prob is not None:
        # PR-AUC is the most robust metric for heavily imbalanced target distributions
        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"PR-AUC  (Precision-Recall Area Under Curve): {pr_auc:.4f}")
        print(f"ROC-AUC (Receiver Operating Characteristic): {roc_auc:.4f}")
        metrics['pr_auc'] = pr_auc
        metrics['roc_auc'] = roc_auc
        
    return metrics

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