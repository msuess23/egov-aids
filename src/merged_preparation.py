import pandas as pd
import numpy as np
import os

def remove_logical_errors(df: pd.DataFrame, columns: list, min_val: float = 0.0) -> pd.DataFrame:
    """
    Drops rows where values in specified columns fall below a logical minimum 
    (e.g., time or money cannot be negative). Critical before log transformations.
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


def remove_extreme_outliers(df: pd.DataFrame, columns: list, upper_quantile: float = 0.999) -> pd.DataFrame:
    """
    Removes extreme outliers by dropping rows that fall above a specified 
    high quantile (e.g., 99.9th percentile).
    """
    print(f"Applying quantile trimming (Dropping top {100 - upper_quantile*100:.2f}%)...")
    initial_rows = len(df)
    valid_cols = [c for c in columns if c in df.columns]
    
    mask = pd.Series(True, index=df.index)
    for col in valid_cols:
        cutoff_val = df[col].quantile(upper_quantile)
        col_mask = (df[col] <= cutoff_val) | (df[col].isna())
        mask = mask & col_mask
        
    df_cleaned = df[mask].copy()
    dropped_count = initial_rows - len(df_cleaned)
    
    print(f" -> Dropped {dropped_count:,} rows containing extreme upper outliers.")
    return df_cleaned


def winsorize_features(df: pd.DataFrame, columns: list, upper_quantile: float = 0.99) -> pd.DataFrame:
    """
    Caps extreme values in input features at a specified quantile instead of dropping.
    """
    print(f"Winsorizing features (Capping at the {upper_quantile*100:.1f}th percentile)...")
    valid_cols = [c for c in columns if c in df.columns]
    
    for col in valid_cols:
        cutoff_val = df[col].quantile(upper_quantile)
        capped_count = (df[col] > cutoff_val).sum()
        df[col] = df[col].clip(upper=cutoff_val)
        
        if capped_count > 0:
            print(f"  -> {col}: Capped {capped_count:,} extreme values at {cutoff_val:,.2f}.")
            
    return df


def apply_log1p_transformation(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies a natural logarithm transformation (log(1+x)) to right-skewed 
    numerical and financial features. Overwrites the original columns.
    """
    print("Applying log1p transformation (Overwriting original columns)...")
    valid_cols = [c for c in columns if c in df.columns]
    
    for col in valid_cols:
        # Overwrite the column in-place
        df[col] = np.log1p(df[col])
        print(f"  -> Log-transformed: {col}")
        
    return df

def encode_categorical_features(df: pd.DataFrame, columns: list, mapping_file: str = "reports/encoding.md") -> pd.DataFrame:
    """
    Converts categorical text columns into integer-based IDs (Label Encoding) 
    and generates a Markdown reference file for traceability (reverting).
    
    Args:
        df (pd.DataFrame): The dataset.
        columns (list): List of column names to encode.
        mapping_file (str): Path to the markdown file where mappings are saved.
        
    Returns:
        pd.DataFrame: Dataframe with integer-encoded categorical features.
    """
    print(f"Encoding categorical features and generating reference: {mapping_file}...")
    valid_cols = [c for c in columns if c in df.columns]
    
    # Ensure the directory for the report exists
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    
    # Start building the Markdown content
    mapping_content = "# Categorical Encoding Reference\n\n"
    mapping_content += "This file contains the mapping between original text values and their integer-encoded IDs.\n\n"

    for col in valid_cols:
        # 1. Convert to pandas category to establish a stable order
        df[col] = df[col].astype('category')
        categories = df[col].cat.categories
        
        # 2. Build the Markdown table for this feature
        mapping_content += f"## Feature: {col}\n"
        mapping_content += "| Encoded ID | Original Value |\n"
        mapping_content += "| :--- | :--- |\n"
        
        for i, original_value in enumerate(categories):
            mapping_content += f"| {i} | {original_value} |\n"
        mapping_content += "\n"
        
        # 3. Apply the encoding (Codes are 0, 1, 2... and -1 for NaNs)
        codes = df[col].cat.codes
        
        # 4. Restore NaNs: XGBoost handles np.nan better than -1
        df[col] = np.where(codes == -1, np.nan, codes)
        
        print(f"  -> {col}: Encoded {len(categories)} categories.")

    # Write the mapping file
    with open(mapping_file, "w", encoding="utf-8") as f:
        f.write(mapping_content)
        
    print(f" -> Encoding complete. Mapping saved to '{mapping_file}'.")
    return df

def impute_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values using targeted strategies:
    - Median for numeric inputs (< 5% missing)
    - Mode for categorical/binary inputs
    - Flagging for variables with massive missingness (> 40%)
    """
    print("Imputing missing values for input features...")
    
    # 1. Median Imputation (Numeric)
    median_cols = ['DURATION', 'PREPARATION_DAYS', 'NUTS_LEVEL']
    for col in median_cols:
        if col in df.columns:
            med_val = df[col].median()
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(med_val)
            if missing_count > 0:
                print(f"  -> {col}: Filled {missing_count:,} NaNs with Median ({med_val:.2f})")

    # 2. Mode Imputation (Categorical/Binary)
    mode_cols = ['B_EU_FUNDS', 'B_RECURRENT_PROCUREMENT', 'TOP_TYPE']
    for col in mode_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            missing_count = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            if missing_count > 0:
                print(f"  -> {col}: Filled {missing_count:,} NaNs with Mode ({mode_val})")

    # 3. Explicit Missing Flag for massive gaps
    if 'ESTIMATED_VALUE_EUR' in df.columns:
        # Create a flag: 1 if missing, 0 if not
        df['ESTIMATED_VALUE_MISSING'] = df['ESTIMATED_VALUE_EUR'].isna().astype(int)
        print("  -> ESTIMATED_VALUE_EUR: Kept NaNs for XGBoost, but generated 'ESTIMATED_VALUE_MISSING' flag.")

    return df