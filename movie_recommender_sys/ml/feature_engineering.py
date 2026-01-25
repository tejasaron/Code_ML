import ast
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from config import MAX_FEATURES, ARTIFACT_DIR


def parse_json(text):
    try:
        return [item["name"] for item in ast.literal_eval(text)]
    except Exception:
        return []

def extract_top_cast(text, top_n=3):
    try:
        cast = ast.literal_eval(text)
        return [member["name"] for member in cast[:top_n]]
    except Exception:
        return []

def extract_director(text):
    try:
        crew = ast.literal_eval(text)
        for member in crew:
            if member.get("job") == "Director":
                return [member["name"]]
        return []
    except Exception:
        return []


def build_features(df):
    """
    Input:
        df: merged movies + credits dataframe

    Output:
        feature_matrix
        movies_metadata (movie_id, title)
    """

    df = df.copy()
    df.dropna(inplace=True)

    # Parse structured columns
    df["genres"] = df["genres"].apply(parse_json)
    df["keywords"] = df["keywords"].apply(parse_json)
    df["cast"] = df["cast"].apply(extract_top_cast)
    df["crew"] = df["crew"].apply(extract_director)

    # Clean overview
    df["overview"] = df["overview"].fillna("").apply(lambda x: x.split())

    # Build tags
    df["tags"] = (
        df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"]
        + df["overview"]
    )

    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

    # Metadata for frontend / TMDB
    movies_metadata = df[["movie_id", "title"]].reset_index(drop=True)

    # Save movies metadata
    movies_path = ARTIFACT_DIR / "movies.pkl"

    with open(movies_path, "wb") as f:
        pickle.dump(movies_metadata, f)

    # Vectorization
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english"
    )

    feature_matrix = vectorizer.fit_transform(df["tags"])

    return feature_matrix
