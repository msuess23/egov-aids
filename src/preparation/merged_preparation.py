import pandas as pd
import numpy as np
import os

def remove_logical_errors(df: pd.DataFrame, columns: list, min_val: float = 0.0) -> pd.DataFrame:
    """
    Removes records with impossible negative values in temporal or financial attributes.
    Procurement data frequently contains entry errors where durations or values are 
    negative. These must be purged before any logarithmic or statistical analysis.
    """
    print(f"Removing logical errors (Dropping values < {min_val})...")
    initial_rows = len(df)
    valid_cols = [c for c in columns if c in df.columns]
    
    mask = pd.Series(True, index=df.index)
    for col in valid_cols:
        # Keep rows where the value is >= min_val OR is NaN
        col_mask = (df[col] >= min_val) | (df[col].isna())
        mask = mask & col_mask
        
    df_cleaned = df[mask].copy()
    dropped_count = initial_rows - len(df_cleaned)
    
    print(f" -> Dropped {dropped_count:,} rows with impossible negative values.")
    return df_cleaned


def impute_domain_specific_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills non-critical missing values using domain-specific logical constants 
    (Missing Not At Random - MNAR).
    This is executed prior to the ML splits to ensure these categories are 
    available for Exploratory Data Analysis (EDA) and profiling. Since these 
    are constants, there is zero risk of data leakage.
    """
    print("Applying domain-specific logic to missing values...")
    df_clean = df.copy()
    
    # 1. The Null Hypothesis: Empty fields in bureaucratic forms generally imply 'No'
    zero_fill_cols = ['B_EU_FUNDS', 'B_RECURRENT_PROCUREMENT']
    for col in zero_fill_cols:
        if col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            df_clean[col] = df_clean[col].fillna(0)
            print(f"  -> {col}: Filled {missing_count:,} NaNs with 0 (No).")

    # 2. The 'Undisclosed' Category: Missing criteria is a distinct procurement strategy
    missing_fill_cols = ['CRIT_CODE', 'AWARD_CRITERION']
    for col in missing_fill_cols:
        if col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                # Handle Pandas Categorical Dtype safety
                if hasattr(df_clean[col], 'cat') and 'MISSING' not in df_clean[col].cat.categories:
                    df_clean[col] = df_clean[col].cat.add_categories(['MISSING'])
                
                df_clean[col] = df_clean[col].fillna('MISSING')
                print(f"  -> {col}: Filled {missing_count:,} NaNs with 'MISSING'.")

    # 3. The Mode Assumption: Safe for variables with extremely low missingness (e.g., < 1%)
    if 'TOP_TYPE' in df_clean.columns:
        missing_count = df_clean['TOP_TYPE'].isna().sum()
        if missing_count > 0:
            mode_val = df_clean['TOP_TYPE'].mode()[0]
            df_clean['TOP_TYPE'] = df_clean['TOP_TYPE'].fillna(mode_val)
            print(f"  -> TOP_TYPE: Filled {missing_count:,} NaNs with mode '{mode_val}'.")
        
    return df_clean


def _calculate_log_iqr_threshold(series: pd.Series, k_factor: float = 3.0) -> float:
    """
    Calculates the outlier threshold in a log-transformed space.
    Financial data and tender counts scale across multiple orders of magnitude (heavy tails).
    A standard linear IQR would incorrectly classify legitimate large-scale projects as 
    outliers. Logarithmic scaling compresses these magnitudes, allowing the IQR 
    to identify only extreme data entry errors (e.g., values in the trillions).
    """
    log_series = np.log1p(series.dropna())
    q1 = log_series.quantile(0.25)
    q3 = log_series.quantile(0.75)
    iqr = q3 - q1
    log_threshold = q3 + (k_factor * iqr)
    return np.expm1(log_threshold)

def _calculate_standard_iqr_threshold(series: pd.Series, k_factor: float = 3.0) -> float:
    """
    Calculates the outlier threshold in linear space.
    Used for variables that scale linearly (e.g., preparation days, region counts). 
    These attributes lack the exponential spread of financial data, making a 
    standard Tukey IQR sufficient for outlier detection.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q3 + (k_factor * iqr)


def remove_extreme_outliers(df: pd.DataFrame, columns: list, k_factor: float = 3.0, method: str = 'log') -> pd.DataFrame:
    """
    Removes extreme outliers (Trimming) based on mathematically derived boundaries.
    Trimming is applied to target variables to ensure that the ground truth for 
    training is not corrupted by severe data entry errors.
    """
    print(f"Applying mathematical trimming ({method.capitalize()}-IQR, k={k_factor})...")
    initial_rows = len(df)
    valid_cols = [c for c in columns if c in df.columns]
    
    mask = pd.Series(True, index=df.index)
    for col in valid_cols:
        if method == 'log':
            cutoff_val = _calculate_log_iqr_threshold(df[col], k_factor)
        else:
            cutoff_val = _calculate_standard_iqr_threshold(df[col], k_factor)
            
        col_mask = (df[col] <= cutoff_val) | (df[col].isna())
        mask = mask & col_mask
        
    df_cleaned = df[mask].copy()
    dropped_count = initial_rows - len(df_cleaned)
    print(f" -> Dropped {dropped_count:,} rows exceeding mathematical {method.capitalize()}-IQR boundaries.")
    return df_cleaned


def winsorize_features(df: pd.DataFrame, columns: list, k_factor: float = 3.0, method: str = 'log') -> pd.DataFrame:
    """
    Caps extreme values (Winsorizing) in input features.
    Capping preserves the record while limiting the influence of extreme values on 
    model training (e.g., tree splits). This prevents the model from overfitting 
    on statistical anomalies while maintaining a high sample size.
    """
    print(f"Winsorizing features (Capping via {method.capitalize()}-IQR, k={k_factor})...")
    valid_cols = [c for c in columns if c in df.columns]
    
    for col in valid_cols:
        if method == 'log':
            cutoff_val = _calculate_log_iqr_threshold(df[col], k_factor)
        else:
            cutoff_val = _calculate_standard_iqr_threshold(df[col], k_factor)
            
        capped_count = (df[col] > cutoff_val).sum()
        df[col] = df[col].clip(upper=cutoff_val)
        
        if capped_count > 0:
            print(f"  -> {col}: Capped {capped_count:,} values at mathematically derived max ({cutoff_val:,.2f}).")
            
    return df


def drop_missing_targets(df: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """
    Purges records missing critical labels.
    Machine learning requires verified outcomes (ground truth) for training and evaluation.
    Records without tender counts or award values are analytically unusable.
    """
    print(f"Dropping rows with missing critical target variables: {target_columns}...")
    initial_rows = len(df)
    df_cleaned = df.dropna(subset=target_columns).copy()
    dropped_count = initial_rows - len(df_cleaned)
    print(f" -> Dropped {dropped_count:,} unusable rows.")
    return df_cleaned