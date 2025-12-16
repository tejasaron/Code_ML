import pandas as pd
from pathlib import Path

def load_raw_data(data_dir):
    '''
    Load all raw energy consumption CSV files and return a merged Dataframe
    '''

    data_dir = Path(data_dir) # 🔑 convert str → Path

    dataframes = []

    for csv_file in data_dir.glob('*.csv'):

        df = pd.read_csv(csv_file)  # 🔑 pass the variable, not a string

        # Required base column
        if "Datetime" not in df.columns:
             raise ValueError(
                 f"Missing 'Datetime' column in {csv_file.name}. "
                 f"Found columns: {list(df.columns)}"
                 )
        
        # Check if ANY MW column exists
        mw_cols = [c for c in df.columns if c.endswith("_MW")]

        if not mw_cols:
            raise ValueError(
                f"No MW columns found in {csv_file.name}. "
                f"Found columns: {list(df.columns)}"
                )
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['utility'] = csv_file.stem.replace('_hourly','') # type: ignore

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.sort_values("Datetime")

    return merged_df

if __name__ == '__main__':
    folder_path = "D:/Code_ML/energy_consumption_forecasting/src/data/raw"
    df = load_raw_data(folder_path)
    print(df.head())
    print(df.info())
