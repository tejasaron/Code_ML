import pandas as pd
from pathlib import Path

def load_raw_data(data_dir):
    '''
    Load all raw energy consumption CSV files and return a merged Dataframe
    '''

    data_dir = Path(data_dir) # 🔑 convert str → Path

    utility_data = {}
    
    for csv_file in data_dir.glob('*.csv'):

        utility_name = csv_file.stem.replace("_hourly", "")

        df = pd.read_csv(csv_file)  # 🔑 pass the variable, not a string

        df.columns = [col.split("_", 1)[1] if col.endswith("_MW") else col 
                      for col in df.columns ]

        # Required base column
        if "Datetime" not in df.columns:
             raise ValueError(
                 f"Missing 'Datetime' column in {csv_file.name}. "
                 f"Found columns: {list(df.columns)}"
                 )
        
                
        df['Datetime'] = pd.to_datetime(df['Datetime'])

        df = df.sort_values('Datetime')
        utility_data[utility_name] = df

    return utility_data

if __name__ == '__main__':
    folder_path = "D:/Code_ML/energy_consumption_forecasting/src/data/raw"
    data = load_raw_data(folder_path)

    for utility, df in data.items():
        print(f"{utility}: {df.shape}")
        print(df.columns)