# ==============================================================================
# src/merging.py (Master Merge, Consistency Check & ML Dataset Prep)
# ==============================================================================
import pandas as pd
import numpy as np

def perform_master_merge(df_cfc: pd.DataFrame, df_can: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the final Inner Join between CFC and CAN using ID_LOT.
    """
    print("Performing final Master Merge (Strict Inner Join on ID_LOT)...")
    
    # 1. Normalize IDs
    def normalize_id(series):
        return series.astype(str).str.strip().str.upper().str.replace(r'\.0$', '', regex=True)
    
    cfc = df_cfc.copy()
    can = df_can.copy()
    
    # Setup join keys for CFC
    cfc['JOIN_NOTICE'] = normalize_id(cfc['FUTURE_CAN_ID'])
    cfc['JOIN_LOT'] = normalize_id(cfc['ID_LOT'].fillna("NO_LOT_DEFINED"))
    
    # Setup join keys for CAN
    can['JOIN_NOTICE'] = normalize_id(can['ID_NOTICE_CAN'])
    can['JOIN_LOT'] = normalize_id(can['ID_LOT'].fillna("NO_LOT_DEFINED"))
    
    # -------------------------------------------------------------------------
    # STRICT MERGE: Structural ID_LOT
    # -------------------------------------------------------------------------
    df_master = pd.merge(
        cfc, can, 
        on=['JOIN_NOTICE', 'JOIN_LOT'], 
        how='inner',
        suffixes=('_CFC', '_CAN')
    )
    
    print(f"Master Merge Complete! Total Rows: {len(df_master):,}")
    return df_master


def compare_feature_consistency(df_master: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Compares the values of shared features between the CFC and CAN datasets.
    Safely handles 'category' dtypes to prevent category mismatch errors.
    """
    print("Calculating Feature Consistency Rates...")
    results = []
    
    for feat in features:
        col_cfc = f"{feat}_CFC"
        col_can = f"{feat}_CAN"
        
        # Check if both columns exist in the merged dataframe
        if col_cfc in df_master.columns and col_can in df_master.columns:
            
            # Convert to generic 'object' to bypass strict Categorical comparisons
            s_cfc = df_master[col_cfc].astype(object)
            s_can = df_master[col_can].astype(object)
            
            # Match condition: Values are identical OR both are NaN
            mask_identical = (s_cfc == s_can)
            mask_both_na = df_master[col_cfc].isna() & df_master[col_can].isna()
            
            match_count = (mask_identical | mask_both_na).sum()
            total_count = len(df_master)
            
            results.append({
                'Feature': feat,
                'Match_Count': match_count,
                'Total_Count': total_count,
                'Consistency_Rate (%)': round((match_count / total_count) * 100, 2)
            })
        else:
            print(f"Warning: Columns for {feat} not found in master dataset.")
            
    # Return sorted DataFrame
    return pd.DataFrame(results).sort_values(by='Consistency_Rate (%)', ascending=False)


def prepare_final_ml_dataset(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the master dataset by selecting specific targets and features.
    Data cleaning (dropping NaNs) and feature engineering are deferred to the 
    subsequent preparation and ML pipelines to strictly separate concerns.
    """
    print("\nPreparing final ML Dataset (Selecting Features & Targets)...")
    
    # 1. Define what to keep
    meta_cols = [
        'JOIN_NOTICE',
        'JOIN_LOT'
    ]

    targets_can = [
        'NUMBER_OF_TENDERS', 
        'TOTAL_VALUE_EUR_CAN',
        'LOT_AWARD_VALUE_EUR'
    ]
    
    features_cfc = [
        'YEAR_CFC',
        'ID_TYPE_CFC',
        'ISO_COUNTRY_CODE_CFC',
        'CAE_TYPE_CFC',
        'MAIN_ACTIVITY_CFC',
        'CPV_CFC',
        'LOTS_NUMBER_CFC',
        'B_EU_FUNDS_CFC',
        'DURATION',
        'TOP_TYPE_CFC',
        'B_RECURRENT_PROCUREMENT',
        'TOTAL_VALUE_EUR_CFC',
        'IS_FRAMEWORK_CFC',
        'COOPERATIVE_PURCHASING_CFC',
        'CRIT_CODE_CFC',
        'TYPE_OF_CONTRACT_CFC',
        'NUTS_REGION_COUNT_CFC',
        'NUTS_LEVEL_CFC',
        'PREPARATION_DAYS'
    ]
    
    # 2. Select only these columns
    cols_to_keep = [c for c in meta_cols + targets_can + features_cfc if c in df_master.columns]
    df_ml = df_master[cols_to_keep].copy()
    
    # 3. Rename columns explicitly for the ML phase
    rename_dict = {col: col.replace('_CFC', '') for col in features_cfc}
    rename_dict['TOTAL_VALUE_EUR_CAN'] = 'TARGET_AWARD_VALUE_EUR'
    rename_dict['TOTAL_VALUE_EUR_CFC'] = 'ESTIMATED_VALUE_EUR'
    rename_dict['JOIN_NOTICE'] = 'ID_NOTICE'
    rename_dict['JOIN_LOT'] = 'ID_LOT'
    
    df_ml = df_ml.rename(columns=rename_dict)

    # 4. Compress high-cardinality Text-IDs into Integers for Report
    df_ml['ID_NOTICE'], _ = pd.factorize(df_ml['ID_NOTICE'])
    df_ml['ID_LOT'], _ = pd.factorize(df_ml['ID_LOT'])
    
    print(f" -> Merged ML Dataset shape: {df_ml.shape}")
    
    return df_ml