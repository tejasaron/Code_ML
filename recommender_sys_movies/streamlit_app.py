import sys
from pathlib import Path
import random

import streamlit as st
import requests

# Adds the project root (recommender_sys_movies/) to the path
root_path = Path("recommender_sys_movies").resolve().parent.parent
sys.path.append(str(root_path))

from src.recommend import recommend, random_movie_list  # your existing functions

TMDB_API_KEY = "a93c7c8dd53536ec8e0db47478a139ce"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# -----------------------------
# Streamlit page config (must be near the top)
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# App-wide CSS to "beautify" UI
# -----------------------------
st.markdown(
    """
<style>
/* --- Page background --- */
.stApp {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(255,255,255,0.06), transparent 60%),
              linear-gradient(180deg, #0b0f14 0%, #070a0f 100%);
  color: #e8eef6;
}

/* --- Main container width & spacing --- */
.block-container {
  padding-top: 2rem;
  max-width: 1200px;
}

/* --- Typography --- */
h1, h2, h3, h4 {
  letter-spacing: -0.02em;
}
h1 {
  margin-bottom: 0.4rem;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1722 0%, #0b111a 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
  color: #dbe7ff !important;
}

/* --- Buttons --- */
.stButton > button {
  width: 100%;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.06);
  color: #eaf2ff;
  padding: 0.65rem 0.9rem;
  transition: transform 120ms ease, background 120ms ease, border 120ms ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.16);
}

/* --- Selectbox --- */
div[data-baseweb="select"] > div {
  border-radius: 12px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
}

/* --- Movie card wrapper --- */
.movie-card {
  padding: 0.35rem 0.35rem 0.8rem 0.35rem;
  border-radius: 16px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  box-shadow: 0 10px 30px rgba(0,0,0,0.30);
  transition: transform 120ms ease, border 120ms ease, background 120ms ease;
}
.movie-card:hover {
  transform: translateY(-3px);
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
}
.movie-title {
  font-size: 0.95rem;
  font-weight: 700;
  margin: 0.55rem 0.4rem 0.35rem 0.4rem;
  line-height: 1.2;
  min-height: 2.4em; /* keeps card titles aligned */
}

/* --- Make Streamlit images nicely rounded --- */
img {
  border-radius: 14px !important;
}

/* Reduce Streamlit default padding around columns */
div[data-testid="column"] {
  padding: 0.4rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Poster fetch (cached) so TMDB isn't called repeatedly on reruns
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int | str) -> str | None:
    """Fetch TMDB poster URL for a movie_id (cached)."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
    except Exception:
        return None

    poster_path = data.get("poster_path")
    if poster_path:
        return TMDB_IMAGE_BASE + poster_path
    return None

# -----------------------------
# Random movies helper
# Your random_movie_list() currently returns a DataFrame (since you use ['title'])
# We wrap it so we always get a "fresh" random set, stable per Shuffle click.
# -----------------------------
def get_random_movies_df(k: int = 6):
    """
    Returns a DataFrame of random movies.
    Assumes random_movie_list(k) returns a DF with columns: ['title', 'movie_id' or 'id'].
    """
    df = random_movie_list(k=k)

    # normalize movie id column so the rest of the app is consistent
    if "movie_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "movie_id"})

    # keep only what we need
    cols = [c for c in ["title", "movie_id"] if c in df.columns]
    return df[cols].dropna().drop_duplicates().reset_index(drop=True)

# -----------------------------
# Header
# -----------------------------
st.title("Movie Recommendation System")
st.caption("Pick a seed movie from the sidebar and get 6 recommendations.")

# -----------------------------
# Sidebar controls (cleaner than a left column)
# -----------------------------
with st.sidebar:
    st.markdown("## 🎲 Random Pick")
    st.caption("Shuffle to get fresh random movies. Select one to recommend.")

    # Session state init: keep the same 6 movies until user presses Shuffle
    if "random_movies_df" not in st.session_state:
        st.session_state.random_movies_df = get_random_movies_df(k=6)

    if st.button("Shuffle"):
        st.session_state.random_movies_df = get_random_movies_df(k=6)

    # Dropdown shows only titles (nice UX)
    random_titles = st.session_state.random_movies_df["title"].tolist()

    movie_name = st.selectbox(
        "Choose a movie",
        random_titles,
        index=0 if random_titles else None,
        key="movie_dropdown",
    )

# -----------------------------
# Main content: recommendations grid
# -----------------------------
if movie_name:
    # Optional: show a spinner while computing recommendations/posters
    with st.spinner("Finding recommendations..."):
        recommendations = recommend(movie_name)  # expected list of dicts with title + movie_id

    st.markdown("### Recommended for you")

    # Show 6 recommendations in a 3-column responsive grid
    cols = st.columns(3)
    for i, rec in enumerate(recommendations[:6]):
        col = cols[i % 3]

        with col:
            poster = fetch_poster(rec["movie_id"])

            # Smaller, cleaner title (instead of huge subheader blocks)
            st.markdown(f'<div class="movie-title">{rec["title"]}</div>', unsafe_allow_html=True)

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.info("Poster not available")

            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Select a movie from the sidebar to see recommendations.")
