from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

def train_xgboost_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Trains an XGBoost Classifier for binary outcomes (e.g., Single Bidding).
    """
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    print(" -> Classifier training complete.")
    return model

def evaluate_classifier(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates a classification model using Accuracy, Precision, Recall, and F1-Score.
    Also plots a normalized Confusion Matrix.
    """
    print("\nEvaluating Classifier Performance...")
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"--- Classification Metrics ---")
    print(f"Accuracy:  {acc:.4f} (Overall correctness)")
    print(f"Precision: {prec:.4f} (When predicted positive, how often correct?)")
    print(f"Recall:    {rec:.4f} (How many actual positives were found?)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean of Precision & Recall)")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, preds, normalize='true')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Normal', 'Single Bidding'], yticklabels=['Normal', 'Single Bidding'])
    plt.title("Confusion Matrix (Normalized by True Class)")
    plt.ylabel("True Reality")
    plt.xlabel("Model Prediction")
    plt.tight_layout()
    plt.show()
    
    return preds