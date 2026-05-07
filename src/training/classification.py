import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.training.preparation import build_preprocessor

def prepare_supervised_data(df: pd.DataFrame, target_col: str, group_col: str, feature_cols: list, test_size: float = 0.2):
    """
    Splits the dataset into training and testing subsets while respecting hierarchical boundaries.
    Procurement lots belonging to the same notice share identical macro-level features. 
    A standard random split would distribute highly correlated twin-lots across both train 
    and test sets, causing data leakage and artificially inflated performance metrics. 
    GroupShuffleSplit ensures all lots of a specific notice remain exclusively in one subset.
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    groups = df_clean[group_col]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    
    return X_train, X_test, y_train, y_test, groups_train


def tune_supervised_model(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, 
                          categorical_cols: list, numeric_cols: list, model_name: str):
    """
    Executes hyperparameter tuning using group-aware cross-validation.
    Procurement market failure is heavily imbalanced (healthy competition outweighs monopolies). 
    Cost-sensitive learning parameters (class_weight='balanced' or scale_pos_weight) are 
    enforced to heavily penalize the misclassification of the minority class, forcing the 
    algorithms to learn the pattern of single-bidding rather than defaulting to the majority class.
    """
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    
    # Calculate imbalance ratio for XGBoost
    neg_class_count = (y_train == 0).sum()
    pos_class_count = (y_train == 1).sum()
    imbalance_ratio = neg_class_count / pos_class_count if pos_class_count > 0 else 1
    
    models = {
        'logreg': {
            'estimator': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'params': {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
        },
        'dt': {
            'estimator': DecisionTreeClassifier(class_weight='balanced', random_state=42),
            'params': {
                'classifier__max_depth': [5, 10, 15, 20],
                'classifier__min_samples_leaf': [10, 50, 100]
            }
        },
        'xgboost': {
            'estimator': XGBClassifier(
                scale_pos_weight=imbalance_ratio, 
                eval_metric='logloss', 
                random_state=42
            ),
            'params': {
                # Increasing the number of sequential trees while slowing down the learning rate
                # allows the gradient boosting process to converge more smoothly on complex patterns.
                'classifier__n_estimators': [100, 300, 500],
                'classifier__learning_rate': [0.05, 0.1],
                
                # Allowing deeper non-linear interactions between variables 
                # (e.g., Sector X Country X Duration X Value).
                'classifier__max_depth': [5, 7, 9],
                
                # Regularization parameters to prevent dominant features (e.g., country codes) 
                # from overshadowing underlying structural procurement features.
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
        }
    }
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', models[model_name]['estimator'])
    ])
    
    cv = GroupKFold(n_splits=3)
    
    search = GridSearchCV(
        pipeline, 
        param_grid=models[model_name]['params'], 
        cv=cv, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_