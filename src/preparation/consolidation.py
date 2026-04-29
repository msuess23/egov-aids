import pandas as pd

def consolidate_cascade_feature(df: pd.DataFrame, target_col: str, cascade_cols: list) -> pd.DataFrame:
    """
    Creates a single consolidated feature using a fallback cascade logic.
    Addresses missing values by sequentially parsing a prioritized list of columns.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The name of the new consolidated column.
        cascade_cols (list): Ordered list of column names to use for fallback.
        
    Returns:
        pd.DataFrame: Dataframe with the new feature and original columns removed.
    """
    valid_cols = [c for c in cascade_cols if c in df.columns]
    
    if not valid_cols:
        return df
        
    df[target_col] = df[valid_cols[0]]
    
    for col in valid_cols[1:]:
        df[target_col] = df[target_col].fillna(df[col])
        
    df = df.drop(columns=valid_cols)
    
    valid_counts = df[target_col].notna().sum()
    print(f" -> Consolidated cascade feature '{target_col}'. Valid values: {valid_counts:,}")
    
    return df

def consolidate_logical_or_feature(df: pd.DataFrame, target_col: str, source_cols: list) -> pd.DataFrame:
    """
    Consolidates multiple binary indicator columns into a single binary feature using logical OR.
    Treats 'Y', 1, and True as positive indicators. Drops source columns afterwards.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The name of the new consolidated column.
        source_cols (list): List of columns to combine.
        
    Returns:
        pd.DataFrame: Dataframe with the consolidated binary feature.
    """
    existing_cols = [c for c in source_cols if c in df.columns]
    
    if not existing_cols:
        return df
        
    # Initialize boolean mask
    mask = pd.Series(False, index=df.index)
    
    # Iteratively apply logical OR
    for col in existing_cols:
        mask = mask | df[col].isin(['Y', 'y', 1, '1', True])
        
    df[target_col] = mask.astype(int)
    df = df.drop(columns=existing_cols)
    
    pos_count = df[target_col].sum()
    print(f" -> Consolidated '{target_col}' via logical OR. Total positive flags: {pos_count:,}")
    
    return df

def consolidate_tenders_and_non_awards(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the total number of tenders and infers zero-tender outcomes based on 
    market failure codes. Drops procedures canceled due to administrative reasons.
    
    Args:
        df (pd.DataFrame): The dataset.
        
    Returns:
        pd.DataFrame: Dataframe with the finalized 'NUMBER_OF_TENDERS' feature.
    """
    if 'INFO_ON_NON_AWARD' in df.columns:
        # Administrative cancellations do not reflect market behavior and are excluded
        mask_discontinued = df['INFO_ON_NON_AWARD'] == 'PROCUREMENT_DISCONTINUED'
        dropped_count = mask_discontinued.sum()
        df = df[~mask_discontinued].copy()
        if dropped_count > 0:
            print(f" -> Excluded {dropped_count:,} rows due to administrative cancellation ('PROCUREMENT_DISCONTINUED').")
    
    tender_cols = ['NUMBER_TENDERS_SME', 'NUMBER_TENDERS_OTHER_EU', 'NUMBER_TENDERS_NON_EU']
    existing_cols = [c for c in tender_cols if c in df.columns]
    
    if existing_cols:
        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df['NUMBER_OF_TENDERS'] = df[existing_cols].sum(axis=1, min_count=1)
        df = df.drop(columns=existing_cols)
        print(f" -> Aggregated 'NUMBER_OF_TENDERS' and dropped sub-category columns.")
    elif 'NUMBER_OF_TENDERS' not in df.columns:
        df['NUMBER_OF_TENDERS'] = np.nan 

    if 'INFO_ON_NON_AWARD' in df.columns:
        # Procedures failing due to lack of suitable offers indicate zero valid tenders
        mask_unsuccessful = df['INFO_ON_NON_AWARD'] == 'PROCUREMENT_UNSUCCESSFUL'
        needs_filling = mask_unsuccessful & df['NUMBER_OF_TENDERS'].isna()
        filled_count = needs_filling.sum()
        
        df.loc[needs_filling, 'NUMBER_OF_TENDERS'] = 0
        df = df.drop(columns=['INFO_ON_NON_AWARD'])
        
        if filled_count > 0:
            print(f" -> Inferred 0 tenders for {filled_count:,} procedures lacking valid offers ('PROCUREMENT_UNSUCCESSFUL').")
            
    valid_counts = df['NUMBER_OF_TENDERS'].notna().sum()
    print(f" -> Finalized 'NUMBER_OF_TENDERS'. Valid target values: {valid_counts:,}")
    
    return df