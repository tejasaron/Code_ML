import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import requests
from st_clickable_images import clickable_images

from ml.recommend import recommend

TMDB_API_KEY = "a93c7c8dd53536ec8e0db47478a139ce"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    r = requests.get(url)
    data = r.json()

    if data.get("poster_path"):
        return TMDB_IMAGE_BASE + data["poster_path"]
    return None


st.title("Movie Recommendation System")

movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    recommendations = recommend(movie_name)

    posters = []
    titles = []

    for rec in recommendations:
        poster = fetch_poster(rec["movie_id"])
        if poster:
            posters.append(poster)
            titles.append(rec["title"])

    # Display-only carousel
    clickable_images(
        posters,
        titles=titles,
        div_style={
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "nowrap",
            "overflow-x": "auto",
        },
        img_style={
            "margin": "8px",
            "height": "220px",
            "border-radius": "12px",
        },
    )
