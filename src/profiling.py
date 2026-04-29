import os
import pandas as pd
from ydata_profiling import ProfileReport

def generate_minimal_report(df: pd.DataFrame, output_path: str, title: str = "Minimal Report") -> None:
    """
    Generates a minimal ydata-profiling report on the full dataset.
    
    Args:
        df (pd.DataFrame): The dataset to profile.
        output_path (str): The full path including filename (e.g., 'data/reports/raw/cfc_minimal.html').
        title (str): Title displayed inside the HTML report.
    """
    print(f"Generating minimal report: '{title}' ...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    profile = ProfileReport(
        df, 
        title=title, 
        minimal=True,
        vars={"cat": {"words": False}, "text": {"words": False}} 
    )
    profile.to_file(output_path)
    print(f"-> Minimal report successfully saved to: {output_path}\n")


def generate_full_sample_report(
    df: pd.DataFrame, 
    output_path: str, 
    title: str = "Full Report", 
    sample_size: int = 250000, 
    random_seed: int = 42
) -> None:
    """
    Generates a comprehensive ydata-profiling report on a random sample of the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to profile.
        output_path (str): The full path including filename (e.g., 'data/reports/clean/can_full.html').
        title (str): Title displayed inside the HTML report.
        sample_size (int): Number of rows to sample.
        random_seed (int): Seed for reproducibility.
    """
    print(f"Generating full report: '{title}' (Sample size: {sample_size:,}) ...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    actual_sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=actual_sample_size, random_state=random_seed)
    
    profile = ProfileReport(
        df_sample, 
        title=title, 
        minimal=False,
        vars={"cat": {"words": False}, "text": {"words": False}}
    )
    profile.to_file(output_path)
    print(f"-> Full report successfully saved to: {output_path}\n")