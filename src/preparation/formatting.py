import pandas as pd
import numpy as np

def convert_yes_no_to_binary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts 'Y'/'N' string indicators into a strictly numeric binary format (1/0)
    for machine learning compatibility. Missing values remain as NaN.
    
    Args:
        df (pd.DataFrame): The dataset.
        columns (list): List of column names to convert.
        
    Returns:
        pd.DataFrame: Dataframe with converted binary columns.
    """
    existing_cols = [c for c in columns if c in df.columns]

    for col in existing_cols:
        df[col] = df[col].astype(str).str.strip().str.upper().map({'Y': 1, 'N': 0})
            
    if existing_cols:
        print(f" -> Converted features to binary (1/0) format: {existing_cols}")
        
    return df

def preprocess_framework_estimation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the algorithmic 'FRA_ESTIMATED' string field into a binary flag 
    to prepare it for standard boolean consolidation.
    """
    if 'FRA_ESTIMATED' in df.columns:
        # EU estimation assumes at least 2 strong indicators (string length >= 2)
        df['B_FRA_ESTIMATED_FLAG'] = (df['FRA_ESTIMATED'].fillna('').astype(str).str.len() >= 2).astype(int)
        df = df.drop(columns=['FRA_ESTIMATED'])
        print(" -> Pre-processed algorithmic 'FRA_ESTIMATED' strings into binary flag.")
        
    return df

def standardize_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Optimizes memory usage and standardizes high-cardinality text fields by 
    converting them to the 'category' data type required for tree-based models.
    
    Args:
        df (pd.DataFrame): The dataset.
        columns (list): List of column names to convert.
        
    Returns:
        pd.DataFrame: Dataframe with optimized categorical columns.
    """
    existing_cols = [c for c in columns if c in df.columns]
    
    for col in existing_cols:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace({'NAN': np.nan, 'NAN ': np.nan, 'NONE': np.nan})
        df[col] = df[col].astype('category')
            
    if existing_cols:
        print(f" -> Standardized {len(existing_cols)} features into categorical format.")
        
    return df

def create_one_hot_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms low-cardinality categorical variables into binary dummy variables 
    to ensure mathematical compatibility with machine learning algorithms.
    
    Args:
        df (pd.DataFrame): The dataset.
        
    Returns:
        pd.DataFrame: Dataframe with new binary dummy columns and source columns removed.
    """
    encoding_map = {
        'TYPE_OF_CONTRACT': {'prefix': 'IS_CONTRACT', 'valid_suffixes': ('U', 'S', 'W')},
        'CRIT_CODE': {'prefix': 'IS_CRIT', 'valid_suffixes': ('L', 'M')}
    }
    
    encoded_count = 0
    
    for col, config in encoding_map.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            dummies = pd.get_dummies(df[col], prefix=config['prefix']).astype(int)
            
            # Keep only valid expected columns to prevent garbage data from creating new features
            valid_cols = [c for c in dummies.columns if c.endswith(config['valid_suffixes'])]
            df = pd.concat([df, dummies[valid_cols]], axis=1)
            df = df.drop(columns=[col])
            
            encoded_count += len(valid_cols)

    if encoded_count > 0:
        print(f" -> Generated {encoded_count} binary one-hot encoded features.")

    return df