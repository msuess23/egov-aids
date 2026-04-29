import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series, is_log_transformed: bool = True):
    """
    Predicts financial/count outcomes and reverses log-transformations 
    to provide metrics in their actual econometric scale (Euros, Bidders).
    """
    print("\\nEvaluating Regression Model...")
    preds_log = model.predict(X_test)
    r2 = r2_score(y_test, preds_log)
    
    # Reverse log1p for interpretability
    y_test_orig = np.expm1(y_test)
    preds_orig = np.expm1(preds_log)
    preds_orig = np.maximum(0, preds_orig) # Physical boundary constraint
    
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))
    mae = mean_absolute_error(y_test_orig, preds_orig)
    
    print(f" -> R² Score (Log-Scale): {r2:.4f}")
    print(f" -> RMSE (Original Scale): {rmse:,.2f}")
    print(f" -> MAE  (Original Scale): {mae:,.2f}")
    return preds_orig

def evaluate_classification(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates binary outcomes (e.g., Single Bidding risk) and plots the Confusion Matrix.
    """
    print("\\nEvaluating Classification Model...")
    preds = model.predict(X_test)
    
    print(f" -> Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f" -> Precision: {precision_score(y_test, preds):.4f}")
    print(f" -> Recall:    {recall_score(y_test, preds):.4f}")
    print(f" -> F1-Score:  {f1_score(y_test, preds):.4f}")
    
    cm = confusion_matrix(y_test, preds, normalize='true')
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Normal', 'Single Bid'], 
                yticklabels=['Normal', 'Single Bid'])
    plt.title("Confusion Matrix (True Rates)")
    plt.ylabel("Actual Reality")
    plt.xlabel("Model Prediction")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names: list, top_n: int = 15, title: str = "Feature Importance"):
    """
    Extracts and visualizes the most influential features. Safely handles 
    sklearn Pipelines by extracting the underlying model step.
    """
    # Extract underlying model if wrapped in an imputation/scaling pipeline
    core_model = model.named_steps['model'] if isinstance(model, Pipeline) else model
    
    # Trees use feature_importances_, Linear models use coef_
    if hasattr(core_model, 'feature_importances_'):
        importance = core_model.feature_importances_
    elif hasattr(core_model, 'coef_'):
        importance = np.abs(core_model.coef_[0] if core_model.coef_.ndim > 1 else core_model.coef_)
    else:
        print("Model does not support feature importance extraction.")
        return

    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_imp['Feature'], df_imp['Importance'], color='#2c3e50')
    plt.title(f"{title} (Top {top_n})")
    plt.xlabel("Relative Importance / Absolute Coefficient")
    plt.tight_layout()
    plt.show()

def analyze_cluster_profiles(df_original: pd.DataFrame, cluster_labels: np.ndarray):
    """
    Calculates the mean values of key features for each identified cluster 
    to assign economic meaning (e.g., 'High-Value Infrastructure' vs 'Local Services').
    """
    df_temp = df_original.copy()
    df_temp['Market_Archetype'] = cluster_labels
    
    # Calculate median values per cluster to avoid heavy outlier skew
    profile = df_temp.groupby('Market_Archetype').median().T
    print("\\nMarket Archetype Profiles (Median Feature Values):")
    display(profile)