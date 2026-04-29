import pandas as pd

def filter_and_reduce(df: pd.DataFrame, dataset_name: str, columns_to_drop: list) -> pd.DataFrame:
    """
    Excludes invalid records (e.g., test files or cancelled procedures) that do not 
    reflect real market interactions, and reduces dimensionality by dropping unused columns.
    
    Args:
        df (pd.DataFrame): The raw dataset.
        dataset_name (str): Identifier for console output (e.g., 'CAN' or 'CFC').
        columns_to_drop (list): List of column names to remove from the dataframe.
        
    Returns:
        pd.DataFrame: The cleaned and reduced dataframe.
    """
    initial_rows = len(df)
    
    # --------------------------------------------------------------------------
    # 1. Row Filtering
    # --------------------------------------------------------------------------
    if 'XSD_VERSION' in df.columns:
        mask_xsd_8 = df['XSD_VERSION'].astype(str).str.contains('8', na=False)
        df = df[~mask_xsd_8]

    if 'CANCELLED' in df.columns:
        df = df[df['CANCELLED'] == 0]
    
    if 'OUT_OF_DIRECTIVES' in df.columns:
        df = df[df['OUT_OF_DIRECTIVES'] == 0]
        
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f" -> [{dataset_name}] Excluded {rows_removed:,} invalid records (e.g., cancelled, out of scope).")

    # --------------------------------------------------------------------------
    # 2. Column Reduction
    # --------------------------------------------------------------------------
    cols_to_drop_existing = [c for c in columns_to_drop if c in df.columns]
    
    if cols_to_drop_existing:
        df = df.drop(columns=cols_to_drop_existing)
        print(f" -> [{dataset_name}] Dropped {len(cols_to_drop_existing)} unneeded columns to reduce dimensionality.")
    
    return df