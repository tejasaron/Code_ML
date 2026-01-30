import sys 
from pathlib import Path

# Adds the project root (recommender_sys_movies/) to the path
root_path = Path('recommender_sys_movies').resolve().parent.parent
sys.path.append(str(root_path))

import streamlit as st
import requests 
from src.recommend import recommend 

TMDB_API_KEY = "a93c7c8dd53536ec8e0db47478a139ce"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500" 

def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    
    response = requests.get(url, params=params)
    
    data = response.json()
    
    if "poster_path" in data and data["poster_path"]:
        return TMDB_IMAGE_BASE + data["poster_path"]
    
    return None

st.title("Movie Recommendation System")

movie_name = st.text_input("Enter a movie name:")


if st.button("Recommend"):
    recommendations = recommend(movie_name)

    rows = [recommendations[:3], recommendations[3:6]]

    for row in rows:

        cols = st.columns(3)

        for col, rec in zip(cols, row):
            poster = fetch_poster(rec["movie_id"])

            with col:
                st.subheader(rec["title"])
                if poster: st.image(poster, use_container_width=True)