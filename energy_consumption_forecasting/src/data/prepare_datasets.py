from pathlib import Path
import sys
PROJECT_ROOT = Path()
sys.path.append(str(PROJECT_ROOT))
print(PROJECT_ROOT)

from src.data import load_data
from src.features.build_features import build_features_for_utility


def prepare_and_save_datasets(raw_dir, processed_dir):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    utility_data = load_data.load_raw_data(raw_dir)
    for utility, df in utility_data.items():
        print(f"Processing utility: {utility}")

        features_df = build_features_for_utility(df)

        output_path = processed_dir / f"{utility}_features.csv"
        features_df.to_csv(output_path, index=False)

        print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    prepare_and_save_datasets(raw_dir="src/data/raw",processed_dir="src/data/processed")