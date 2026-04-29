import pandas as pd
import numpy as np

def calculate_days(df: pd.DataFrame, start_field: str, end_field: str, new_field: str) -> pd.DataFrame:
    """
    Calculates the time span between two dates in days.
    Provides explicit date formatting to ensure parsing consistency.
    
    Args:
        df (pd.DataFrame): The dataset.
        start_field (str): Column name representing the start date.
        end_field (str): Column name representing the end date.
        new_field (str): Column name for the resulting days calculation.
        
    Returns:
        pd.DataFrame: Dataframe with the new numeric days feature.
    """
    if start_field in df.columns and end_field in df.columns:
        # Explicitly define the format (DD/MM/YY) to prevent dateutil fallback warning
        date_format = '%d/%m/%y'
        
        dt_start = pd.to_datetime(df[start_field], format=date_format, errors='coerce')
        dt_end = pd.to_datetime(df[end_field], format=date_format, errors='coerce')
        
        df[new_field] = (dt_end - dt_start).dt.days
        df = df.drop(columns=[start_field, end_field])
        
        valid_counts = df[new_field].notna().sum()
        print(f" -> Calculated '{new_field}'. Valid numeric values: {valid_counts:,}")
    else:
        print(f" -> Warning: Date columns missing. Skipped calculation for '{new_field}'.")
        
    return df

def reduce_cpv_cardinality(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    """
    Reduces the cardinality of the CPV (Common Procurement Vocabulary) feature 
    by extracting the top hierarchical levels (Divisions = 2 digits).
    Handles potential float conversions and restores missing leading zeros.
    
    Args:
        df (pd.DataFrame): The dataset.
        digits (int): The number of top-level hierarchy digits to retain.
        
    Returns:
        pd.DataFrame: Dataframe with reduced CPV feature.
    """
    if 'CPV' not in df.columns:
        print(" -> Warning: 'CPV' column not found. Skipping cardinality reduction.")
        return df

    original_uniques = df['CPV'].nunique()

    # Convert to numeric to strip decimal artifacts (e.g. 45200000.0 -> 45200000)
    cpv_str = pd.to_numeric(df['CPV'], errors='coerce').astype('Int64').astype(str)
    
    valid_mask = cpv_str != '<NA>'
    
    # Restore the 8-digit structure to prevent '03' from being parsed as '3'
    cpv_cleaned = cpv_str[valid_mask].str.zfill(8).str[:digits]
    
    df.loc[valid_mask, 'CPV'] = cpv_cleaned
    df['CPV'] = df['CPV'].astype('category')
    
    new_uniques = df['CPV'].nunique()
    print(f" -> Reduced CPV hierarchical cardinality: {original_uniques:,} -> {new_uniques:,} unique categories.")
    
    return df

def extract_nuts_features(df: pd.DataFrame, target_col: str = 'TAL_LOCATION_NUTS') -> pd.DataFrame:
    """
    Extracts structural metrics (regional count and granularity level) from 
    complex administrative region strings to capture geographic complexity.
    
    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The column containing the comma-separated NUTS codes.
        
    Returns:
        pd.DataFrame: Dataframe with newly extracted spatial features.
    """
    if target_col not in df.columns:
        return df

    nuts_str = df[target_col].fillna('')
    
    # Feature A: Count number of regions
    df['NUTS_REGION_COUNT'] = np.where(nuts_str == '', 0, nuts_str.str.count(',') + 1)

    # Feature B: Granularity / Level
    def get_avg_nuts_level(val):
        if not val: 
            return np.nan
        codes = [len(code.strip()) for code in val.split(',') if len(code.strip()) >= 2]
        if not codes: 
            return np.nan
        return sum(codes) / len(codes) - 2

    df['NUTS_LEVEL'] = nuts_str.apply(get_avg_nuts_level)
    
    df = df.drop(columns=[target_col])
    
    print(" -> Extracted spatial complexity metrics ('NUTS_REGION_COUNT', 'NUTS_LEVEL').")

    return df

def extract_winner_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses concatenated strings representing consortia and extracts absolute counts 
    for SMEs and international participants to assess market composition.
    
    Args:
        df (pd.DataFrame): The dataset (typically the CAN dataset).
        
    Returns:
        pd.DataFrame: Dataframe with new quantitative winner features.
    """
    extracted_features = 0
    
    if 'WIN_COUNTRY_CODE' in df.columns:
        win_str = df['WIN_COUNTRY_CODE'].fillna('')
        df['NUMBER_OF_WINNERS'] = np.where(win_str == '', 0, win_str.str.count('---') + 1)
        
        def count_unique_countries(val):
            if not val: 
                return 0
            codes = set([code.strip() for code in val.split('---') if code.strip()])
            return len(codes)

        df['NUMBER_WINNER_COUNTRIES'] = win_str.apply(count_unique_countries)
        df = df.drop(columns=['WIN_COUNTRY_CODE'])
        extracted_features += 2

    if 'B_CONTRACTOR_SME' in df.columns:
        sme_str = df['B_CONTRACTOR_SME'].fillna('').astype(str).str.upper()
        df['NUMBER_SME_WINNERS'] = sme_str.str.count('Y')
        df = df.drop(columns=['B_CONTRACTOR_SME'])
        extracted_features += 1

    if extracted_features > 0:
        print(f" -> Extracted {extracted_features} quantitative market composition metrics from winner strings.")

    return df