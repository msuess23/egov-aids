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

def plot_feature_importance(pipeline, feature_names_numeric: list, feature_names_categorical: list, top_n: int = 15):
    """
    Extracts and ranks the predictive drivers of the model.
    Transforms the algorithmic rules back into economic insights, answering the core 
    research question regarding which variables trigger single-bidding scenarios.
    Note: Requires tree-based models (DecisionTree, RandomForest, XGBoost).
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    if not hasattr(classifier, 'feature_importances_'):
        print("Model does not support native feature importances.")
        return
        
    ohe_features = []
    if 'cat' in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_['cat']
        ohe_features = list(ohe.get_feature_names_out(feature_names_categorical))
        
    all_features = feature_names_numeric + ohe_features
    importances = classifier.feature_importances_
    
    if len(all_features) != len(importances):
        print("Feature mismatch. Cannot plot importances.")
        return
        
    df_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        hue='Feature', 
        data=df_imp, 
        palette='viridis', 
        legend=False
    )
    plt.title(f"Top {top_n} Predictors for Market Failure")
    plt.xlabel("Relative Feature Importance")
    plt.ylabel("Procurement Feature")
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

def plot_grouped_feature_importance(pipeline, numeric_cols: list, categorical_cols: list):
    """
    Calculates and visualizes the aggregated feature importance for parent categories.
    One-Hot-Encoding fragments the predictive weight of high-cardinality variables 
    (e.g., ISO_COUNTRY_CODE) into multiple low-weight dummy variables. By summing 
    the Gini importances of all derivative columns, the true global impact of the 
    original macro-feature (e.g., Geography vs. Finance) is restored and visualized.
    """
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    if not hasattr(classifier, 'feature_importances_'):
        print("Model does not support native feature importances.")
        return
        
    # Extract One-Hot-Encoded feature names
    ohe_features = []
    if 'cat' in preprocessor.named_transformers_:
        ohe = preprocessor.named_transformers_['cat']
        ohe_features = list(ohe.get_feature_names_out(categorical_cols))
        
    all_features = numeric_cols + ohe_features
    importances = classifier.feature_importances_
    
    if len(all_features) != len(importances):
        print("Feature mismatch. Cannot map importances.")
        return
        
    df_imp = pd.DataFrame({'Feature_OHE': all_features, 'Importance': importances})
    
    # Mapping derivative OHE columns back to their parent feature
    def get_parent_feature(feature_name):
        if feature_name in numeric_cols:
            return feature_name
        for cat in categorical_cols:
            # Matches prefix generated by OneHotEncoder (e.g., "ISO_COUNTRY_CODE_")
            if feature_name.startswith(cat + '_'):
                return cat
        return feature_name

    df_imp['Parent_Feature'] = df_imp['Feature_OHE'].apply(get_parent_feature)
    
    # Aggregating importances by parent feature
    df_grouped = df_imp.groupby('Parent_Feature')['Importance'].sum().reset_index()
    df_grouped = df_grouped.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance', 
        y='Parent_Feature', 
        hue='Parent_Feature', 
        data=df_grouped, 
        palette='magma', 
        legend=False
    )
    plt.title("Aggregated Feature Importance (Macro Level)")
    plt.xlabel("Total Relative Importance (Sum of OHE Variables)")
    plt.ylabel("Parent Feature / Category")
    plt.tight_layout()
    plt.show()