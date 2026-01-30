import sys
from pathlib import Path
# Adds the project root (recommender_sys_movies/) to the path
root_path = Path('recommender_sys_movies').resolve().parent.parent
sys.path.append(str(root_path))



import pickle
import numpy as np
import pandas as pd



from src.config import (
    MOVIES_PATH,
    CREDITS_PATH,
    ARTIFACT_DIR
)

from feature_engineering import build_features
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Load Dataset

    movies_df = pd.read_csv(MOVIES_PATH)
    credits_df = pd.read_csv(CREDITS_PATH)
    
    df = movies_df.merge(credits_df, on="title")

    # Build Features
    
    feature_matrix = build_features(df)

    # Build Similarity

    similarity_matrix = cosine_similarity(feature_matrix)

    # Save Artifacts
    similarity_path = ARTIFACT_DIR / "similarity_matrix.pkl"

    with open(similarity_path, "wb") as f:
        pickle.dump(similarity_matrix, f)

if __name__ == "__main__":
    main()

