import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "ml" / "artifacts"

with open(ARTIFACT_DIR / "movies.pkl", "rb") as f:
    movies_df = pickle.load(f)

with open(ARTIFACT_DIR / "similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

title_to_index = {
    title: idx for idx, title in enumerate(movies_df["title"])
}

index_to_movie = movies_df.to_dict(orient="index")

def recommend(movie_title: str, top_k: int=5):
    if movie_title not in title_to_index:
        return []
    
    movie_idx = title_to_index[movie_title]

    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))

    # Sort by similarity score (descending)

    similarity_scores = sorted(
        similarity_scores,
        key = lambda x: x[1],
        reverse=True
    )

    # Exclude the input movie itself
    similarity_scores = similarity_scores[1: top_k + 1]

    recommendations = []

    for idx, score in similarity_scores:
        movie = index_to_movie[idx]
        recommendations.append({
            "movie_id": movie["movie_id"],
            "title": movie["title"],
            "score": float(score)
        })

    return recommendations


if __name__ == "__main__":
    results = recommend("Avatar", top_k=5)
    print(results)