import pickle
import numpy as np
import random
from src.config import  ARTIFACT_DIR


def load_artifacts():

    movies_path = ARTIFACT_DIR / "movies_metadata.pkl"
    similarity_path = ARTIFACT_DIR / "similarity_matrix.pkl"

    with open(movies_path, "rb") as f:
        movies = pickle.load(f)
    
    with open(similarity_path, "rb") as f:
        similarity_matrix = pickle.load(f)
    
    return movies, similarity_matrix

def random_movie_list(k=5):
    movies, _ = load_artifacts()
    return movies.sample(n=k, replace=False).reset_index(drop=True)

def recommend(movies_title, top_k=6):
    movies, similarity_matrix = load_artifacts()

    # Find index of selected movie
    idx = movies[movies["title"] == movies_title].index[0]

    # Get similarity scores
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity (descending)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip self and take top_k
    scores = scores[1:top_k + 1]

    recommendations = []
    for i, _ in scores:
        recommendations.append({
            "movie_id": int(movies.iloc[i]["movie_id"]),
            "title": movies.iloc[i]["title"]
        })

    return recommendations
