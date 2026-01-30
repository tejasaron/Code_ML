import pickle
import numpy as np

def load_artifacts():
    with open("ml/artifacts/movies.pkl", "rb") as f:
        movies = pickle.load(f)
    
    with open("ml/artifacts/similarity_matrix.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)
    
    return movies, similarity_matrix

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
