import pandas as pd
import os

def convert_csv_to_parquet(csv_path: str, parquet_path: str, dataset_name: str = "Dataset") -> pd.DataFrame:
    """
    Loads a raw CSV file into memory and saves it as a Parquet file.
    
    Args:
        csv_path (str): File path to the raw CSV data.
        parquet_path (str): Destination file path for the Parquet output.
        dataset_name (str, optional): Name of the dataset for console logging. Defaults to "Dataset".
        
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    print(f"[{dataset_name}] Loading CSV file into memory: {csv_path} ...")
    
    # low_memory=False prevents DtypeWarnings for columns with mixed types
    df = pd.read_csv(csv_path, low_memory=False)
    
    rows, cols = df.shape
    print(f"[{dataset_name}] Loaded successfully. Rows: {rows:,} | Columns: {cols}")
    
    print(f"[{dataset_name}] Saving as Parquet format: {parquet_path} ...")
    
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    
    df.to_parquet(parquet_path, index=False)
    print(f"[{dataset_name}] Checkpointing complete. Parquet file is ready.\n")
    
    return df