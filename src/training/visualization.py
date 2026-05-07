import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay

def plot_confusion_matrix_heatmap(y_test: pd.Series, y_pred: np.ndarray):
    """
    Visualizes the absolute counts of True Positives, False Positives, etc.
    Provides immediate context on whether the cost-sensitive model overcompensated 
    (producing too many False Positives for market failure).
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix: Market Failure Prediction")
    plt.xlabel("Predicted Class (0 = Competitive, 1 = Market Failure)")
    plt.ylabel("Actual Class (0 = Competitive, 1 = Market Failure)")
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_test: pd.Series, y_prob: np.ndarray):
    """
    Plots the Precision-Recall Curve.
    Essential for visualizing the trade-off between capturing all market failures (Recall) 
    and ensuring predictions are reliable (Precision) within highly imbalanced datasets.
    """
    plt.figure(figsize=(6, 4))
    display = PrecisionRecallDisplay.from_predictions(y_test, y_prob, name="Model")
    display.ax_.set_title("Precision-Recall Curve (Minority Class Focus)")
    plt.tight_layout()
    plt.show()


def plot_cluster_profiles(df: pd.DataFrame, cluster_col: str, features: list):
    """
    Visualizes cluster characteristics using Boxplots.
    Follows Tufte's principles by using clean axes and high data-to-ink ratio.
    """
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    
    for i, feat in enumerate(features):
        sns.boxplot(data=df, x=cluster_col, y=feat, ax=axes[i], hue=cluster_col, palette="Blues_d", legend=False)
        axes[i].set_title(f'Cluster Dist: {feat}')
        axes[i].set_xlabel('Cluster ID')
        
    plt.tight_layout()
    plt.show()


def plot_regression_density(y_true, y_pred):
    """Uses hexbin density plots to visualize results on large datasets without overplotting."""
    plt.figure(figsize=(8, 6))
    plt.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.colorbar(label='Observation Density')
    plt.title('Regression Fidelity (Pred vs. Actual)')
    plt.xlabel('Actual (Log)')
    plt.ylabel('Predicted (Log)')
    plt.show()


def plot_cluster_scatter_2d(X_processed, labels, vis_sample_size=10000, point_alpha=0.6, point_size=15):
    """
    Projects multi-dimensional cluster data into a 2D space using PCA 
    and visualizes the clusters as a colored scatter plot.
    To avoid Tufte's 'overplotting', a representative sample is drawn.
    """
    valid_mask = labels != -1
    X_valid = X_processed[valid_mask]
    labels_valid = labels[valid_mask]
    
    if len(set(labels_valid)) < 2:
        print("Not enough valid clusters for 2D scatter plot.")
        return
        
    n_total = X_valid.shape[0]
    idx = np.random.choice(n_total, min(vis_sample_size, n_total), replace=False)
    
    X_sample = X_valid[idx]
    
    # Convert sparse to dense if necessary (for PCA)
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()
        
    # Project down to 2 dimensions for plotting
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_2d[:, 0], 
        y=X_2d[:, 1], 
        hue=labels_valid[idx], 
        palette='tab10', 
        alpha=point_alpha,
        s=point_size
    )
    plt.title('2D PCA Projection of Market Clusters')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')
    plt.legend(title='Cluster ID')
    plt.show()

def plot_categorical_profiling_heatmap(df: pd.DataFrame, cluster_col: str, cat_col: str, top_n: int = 10):
    """
    Creates a heatmap showing the concentration of categorical values within clusters.
    Row = Cluster, Column = Category, Color = Percentage of Cluster.
    This is the core tool for Post-Hoc Profiling.
    """
    # Filter for the top N categories to keep the plot readable
    top_cats = df[cat_col].value_counts().nlargest(top_n).index
    df_filtered = df[df[cat_col].isin(top_cats)]
    
    if df_filtered.empty:
        print(f"Not enough data to profile {cat_col}.")
        return

    # Calculate percentages: How much of each cluster belongs to which category?
    crosstab = pd.crosstab(df_filtered[cluster_col], df_filtered[cat_col], normalize='index') * 100
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(crosstab, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Percentage within Cluster (%)'})
    plt.title(f'Post-Hoc Profiling: {cat_col} concentration per Cluster')
    plt.ylabel('Cluster ID')
    plt.xlabel(cat_col)
    
    # Rotate x-labels for better readability of long category names
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def _get_model_and_importances(pipeline):
    """
    Helper function to safely extract the model step and its importances,
    regardless of whether it's a classification or regression task, 
    and whether it's a tree-based or linear model.
    """
    # 1. Identifiziere den Aufgabentyp (Klassifikation vs. Regression)
    if 'classifier' in pipeline.named_steps:
        model_step = pipeline.named_steps['classifier']
    elif 'regressor' in pipeline.named_steps:
        model_step = pipeline.named_steps['regressor']
    else:
        raise KeyError("Pipeline must contain a step named 'classifier' or 'regressor'.")

    # 2. Extrahiere die Wichtigkeit (Bäume vs. Lineare Modelle)
    if hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
    elif hasattr(model_step, 'coef_'):
        # Für lineare Modelle (z.B. Lasso, LogReg) nehmen wir den absoluten Betrag der Koeffizienten
        importances = np.abs(model_step.coef_)
        # Bei manchen Implementierungen (z.B. Multiclass LogReg) ist coef_ 2D. Wir glätten es.
        if importances.ndim > 1:
            importances = importances[0]
    else:
        raise AttributeError("Model has neither 'feature_importances_' nor 'coef_'. Cannot plot.")
        
    return model_step, importances


def plot_feature_importance(pipeline, feature_names_numeric: list, feature_names_categorical: list, top_n: int = 15):
    """
    Extracts and ranks the predictive drivers of the model.
    Transforms the algorithmic rules back into economic insights.
    Works dynamically for both Classifiers and Regressors.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        _, importances = _get_model_and_importances(pipeline)
    except Exception as e:
        print(f"Skipping plot: {e}")
        return

    # Extract dynamic feature names after One-Hot-Encoding
    cat_features = []
    if 'cat' in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_['cat']
        # Depending on the pipeline structure, we might need to access the OHE directly
        if hasattr(cat_transformer, 'named_steps') and 'ohe' in cat_transformer.named_steps:
            ohe = cat_transformer.named_steps['ohe']
        else:
            ohe = cat_transformer
        
        if hasattr(ohe, 'get_feature_names_out'):
            cat_features = list(ohe.get_feature_names_out(feature_names_categorical))

    all_features = feature_names_numeric + cat_features

    # Safety check in case feature counts mismatch
    if len(importances) != len(all_features):
        print(f"Warning: Model has {len(importances)} features, but we extracted {len(all_features)} names. Plotting may be inaccurate.")
        # Fallback to generic names if mismatch occurs
        all_features = [f"Feature {i}" for i in range(len(importances))]

    # Create DataFrame for visualization
    df_imp = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title('Top Predictive Features (Economic Drivers)', fontweight='bold')
    plt.xlabel('Relative Importance / Absolute Coefficient')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()


def plot_grouped_feature_importance(pipeline, feature_names_numeric: list, feature_names_categorical: list):
    """
    Aggregates the importance of one-hot-encoded categorical features back to their 
    original macro-level categories (e.g., combining all CPV_*** dummy variables 
    into one total 'Sector (CPV)' score).
    """
    preprocessor = pipeline.named_steps['preprocessor']
    
    try:
        _, importances = _get_model_and_importances(pipeline)
    except Exception as e:
        print(f"Skipping plot: {e}")
        return

    cat_features = []
    if 'cat' in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'named_steps') and 'ohe' in cat_transformer.named_steps:
            ohe = cat_transformer.named_steps['ohe']
        else:
            ohe = cat_transformer
            
        if hasattr(ohe, 'get_feature_names_out'):
            cat_features = list(ohe.get_feature_names_out(feature_names_categorical))

    all_features = feature_names_numeric + cat_features

    if len(importances) != len(all_features):
        print("Feature length mismatch. Cannot group features reliably.")
        return

    df_imp = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    })

    # Grouping logic: Re-assign dummies to their parent category
    df_imp['Macro_Category'] = df_imp['Feature']
    for cat in feature_names_categorical:
        # Check if the feature name starts with the parent category name
        df_imp.loc[df_imp['Feature'].str.startswith(cat), 'Macro_Category'] = cat

    # Aggregate importance by parent category
    df_grouped = df_imp.groupby('Macro_Category')['Importance'].sum().reset_index()
    df_grouped = df_grouped.sort_values('Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Macro_Category', data=df_grouped, palette='mako')
    plt.title('Macro-Level Feature Importance', fontweight='bold')
    plt.xlabel('Total Aggregated Importance')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()