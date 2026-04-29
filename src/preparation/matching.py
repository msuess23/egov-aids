import pandas as pd

def split_by_notice(df_cfc: pd.DataFrame, df_can: pd.DataFrame) -> dict:
    """
    Partitions the CFC and CAN datasets into matched and orphaned subsets 
    based on their relational Notice IDs.
    
    Args:
        df_cfc (pd.DataFrame): Contract Notices dataset.
        df_can (pd.DataFrame): Contract Award Notices dataset.
        
    Returns:
        dict: Subsets of matched and orphaned dataframes along with match statistics.
    """
    cfc_ids = set(df_cfc.get('FUTURE_CAN_ID', pd.Series(dtype=str)).dropna())
    can_ids = set(df_can.get('ID_NOTICE_CAN', pd.Series(dtype=str)).dropna())
    
    matched_ids = cfc_ids.intersection(can_ids)
    
    mask_cfc = df_cfc.get('FUTURE_CAN_ID', pd.Series(dtype=str)).isin(matched_ids)
    mask_can = df_can.get('ID_NOTICE_CAN', pd.Series(dtype=str)).isin(matched_ids)
    
    cfc_matched, cfc_orphans = df_cfc[mask_cfc].copy(), df_cfc[~mask_cfc].copy()
    can_matched, can_orphans = df_can[mask_can].copy(), df_can[~mask_can].copy()
    
    print(f" -> Notice-Level Match: {len(matched_ids):,} shared IDs identified.")
    print(f"    CFC: {len(cfc_matched):,} matches | {len(cfc_orphans):,} orphans.")
    print(f"    CAN: {len(can_matched):,} matches | {len(can_orphans):,} orphans.")
    
    return {
        'cfc_matched': cfc_matched, 'cfc_orphans': cfc_orphans,
        'can_matched': can_matched, 'can_orphans': can_orphans
    }

def split_by_lot(df_cfc: pd.DataFrame, df_can: pd.DataFrame) -> dict:
    """
    Partitions the lot-level datasets into matched and orphaned subsets 
    using a strict composite key (Notice ID + Lot ID) to ensure relational integrity.
    """
    cfc_check = df_cfc.copy()
    can_check = df_can.copy()
    
    def normalize_id(series):
        return series.astype(str).str.strip().str.upper().str.replace(r'\.0$', '', regex=True)
    
    # Generate composite keys
    cfc_key = normalize_id(cfc_check['FUTURE_CAN_ID']) + "---" + normalize_id(cfc_check['ID_LOT'].fillna("NO_LOT_DEFINED"))
    can_key = normalize_id(can_check['ID_NOTICE_CAN']) + "---" + normalize_id(can_check['ID_LOT'].fillna("NO_LOT_DEFINED"))
    
    cfc_check['COMPOSITE_KEY'] = cfc_key
    can_check['COMPOSITE_KEY'] = can_key
    
    matched_keys = set(cfc_key).intersection(set(can_key))
    
    mask_cfc = cfc_check['COMPOSITE_KEY'].isin(matched_keys)
    mask_can = can_check['COMPOSITE_KEY'].isin(matched_keys)
    
    cfc_matched, cfc_orphans = df_cfc[mask_cfc].copy(), df_cfc[~mask_cfc].copy()
    can_matched, can_orphans = df_can[mask_can].copy(), df_can[~mask_can].copy()
    
    print(f" -> Lot-Level Match: {len(matched_keys):,} strict composite keys aligned.")
    print(f"    CFC: {len(cfc_matched):,} matches | {len(cfc_orphans):,} orphans.")
    print(f"    CAN: {len(can_matched):,} matches | {len(can_orphans):,} orphans.")
    
    return {
        'cfc_matched': cfc_matched, 'cfc_orphans': cfc_orphans,
        'can_matched': can_matched, 'can_orphans': can_orphans
    }

def finalize_datasets(split_results: dict) -> tuple:
    """
    Extracts the finalized matched datasets and discards orphaned records.
    Universally applicable for both notice-level and lot-level pipeline stages.
    """
    cfc_final = split_results['cfc_matched']
    can_final = split_results['can_matched']
    
    print(f" -> Finalized datasets. Orphans discarded.")
    print(f"    CFC Final Rows: {len(cfc_final):,}")
    print(f"    CAN Final Rows: {len(can_final):,}")
    
    return cfc_final, can_final