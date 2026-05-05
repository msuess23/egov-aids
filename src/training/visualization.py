import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix_heatmap(y_true, y_pred):
    """Displays a linear heatmap of Type I and Type II errors."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Classification Error Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
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