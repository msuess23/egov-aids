import pandas as pd

def compare_distributions(df_matched: pd.DataFrame, df_orphans: pd.DataFrame, feature: str, normalize: bool = False) -> pd.DataFrame:
    """
    Compares the categorical or temporal distribution of a specific feature 
    between matched and orphaned datasets to identify systemic drop-out bias.
    
    Args:
        df_matched (pd.DataFrame): The matched dataset.
        df_orphans (pd.DataFrame): The orphaned dataset.
        feature (str): The column name to analyze.
        normalize (bool): If True, returns relative percentages instead of absolute counts.
        
    Returns:
        pd.DataFrame: A formatted comparison table.
    """
    if feature not in df_matched.columns or feature not in df_orphans.columns:
        print(f" -> Warning: Feature '{feature}' missing in datasets. Skipping distribution comparison.")
        return pd.DataFrame()

    val_m = df_matched[feature].value_counts(normalize=normalize).rename('Matched')
    val_o = df_orphans[feature].value_counts(normalize=normalize).rename('Orphans')
    
    comp = pd.concat([val_m, val_o], axis=1).fillna(0)
    
    if normalize:
        comp *= 100
        comp['Difference (%-Points)'] = comp['Orphans'] - comp['Matched']
    else:
        comp['Orphan_Rate (%)'] = (comp['Orphans'] / (comp['Matched'] + comp['Orphans'])) * 100
        
    print(f" -> Analyzed distribution bias for feature '{feature}'.")
    return comp.sort_index().round(2)

def compare_metrics(df_matched: pd.DataFrame, df_orphans: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """
    Computes and compares central tendencies (Median, Mean) for quantitative 
    metrics between matched and orphaned datasets.
    """
    results = []
    existing_metrics = [m for m in metrics if m in df_matched.columns and m in df_orphans.columns]
    
    for metric in existing_metrics:
        results.append({
            'Metric': metric,
            'Matched_Median': df_matched[metric].median(),
            'Orphan_Median': df_orphans[metric].median(),
            'Matched_Mean': df_matched[metric].mean(),
            'Orphan_Mean': df_orphans[metric].mean()
        })
        
    if results:
        print(f" -> Compared central tendencies for {len(existing_metrics)} metric(s).")
        
    return pd.DataFrame(results).round(2)

def analyze_non_award_rates(can_orphans: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates what fraction of CAN orphans represents procedures that were 
    officially closed without an award (market failure or administrative abortion).
    """
    if 'NOT_AWARDED' not in can_orphans.columns:
        return pd.DataFrame()
        
    total = len(can_orphans)
    non_awarded = can_orphans['NOT_AWARDED'].sum()
    rate = (non_awarded / total) * 100 if total > 0 else 0
    
    print(f" -> Analyzed non-award rate among CAN orphans: {rate:.2f}%")
    
    return pd.DataFrame({
        'Metric': ['Total CAN Orphans', 'Non-Awarded Orphans', 'Non-Award Rate (%)'],
        'Value': [total, non_awarded, round(rate, 2)]
    })

def analyze_cross_dataset_orphans(df_cfc: pd.DataFrame, df_can: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies structurally flawed notice definitions by checking if lot orphans 
    in one dataset correspond to completely defined (matched) notices in the other.
    """
    cfc_orphan_notices = set(df_cfc['FUTURE_CAN_ID'].dropna())
    can_matched_notices = set(df_can['ID_NOTICE_CAN'].dropna())
    cfc_cross_orphans = cfc_orphan_notices.intersection(can_matched_notices)
    
    can_orphan_notices = set(df_can['ID_NOTICE_CAN'].dropna())
    cfc_matched_notices = set(df_cfc['FUTURE_CAN_ID'].dropna())
    can_cross_orphans = can_orphan_notices.intersection(cfc_matched_notices)
    
    print(f" -> Identified cross-dataset structural orphans.")
    
    return pd.DataFrame({
        'Direction': ['CFC Lot Orphan in Matched CAN', 'CAN Lot Orphan in Matched CFC'],
        'Affected Notices': [len(cfc_cross_orphans), len(can_cross_orphans)]
    })

def analyze_partial_notice_success(cfc_matched: pd.DataFrame, cfc_orphans: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates whether failing CFC notices failed entirely (no lots awarded) 
    or partially (some lots awarded, some orphaned).
    """
    if 'FUTURE_CAN_ID' not in cfc_matched.columns or 'FUTURE_CAN_ID' not in cfc_orphans.columns:
        return pd.DataFrame()

    total_lots = pd.concat([cfc_matched, cfc_orphans])['FUTURE_CAN_ID'].value_counts().rename('Total_Lots')
    orphan_lots = cfc_orphans['FUTURE_CAN_ID'].value_counts().rename('Orphan_Lots')
    
    df_compare = pd.concat([total_lots, orphan_lots], axis=1).fillna(0)
    df_failing = df_compare[df_compare['Orphan_Lots'] > 0]
    
    total_failures = (df_failing['Total_Lots'] == df_failing['Orphan_Lots']).sum()
    partial_successes = (df_failing['Total_Lots'] > df_failing['Orphan_Lots']).sum()
    
    total_failing_notices = len(df_failing)
    
    print(" -> Analyzed partial vs. total failure scenarios for orphaned notices.")
    
    return pd.DataFrame({
        'Scenario': ['Total Failure (0% Lots Awarded)', 'Partial Success (>0% Lots Awarded)'],
        'Notice Count': [total_failures, partial_successes],
        'Share (%)': [
            round((total_failures / total_failing_notices) * 100, 2) if total_failing_notices > 0 else 0,
            round((partial_successes / total_failing_notices) * 100, 2) if total_failing_notices > 0 else 0
        ]
    })