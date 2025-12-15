import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_data(csv_path):
    '''
    Loads smart grid data and prepares it for analysis.
    '''
    csv_path = PROJECT_ROOT / csv_path
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[()%/]", "", regex=True)
        .str.replace(" ", "_")
    )

    # Convert timestamp
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found. Columns: {df.columns.tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by time (important for forecasting)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df