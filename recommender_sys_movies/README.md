🎬 Movie Recommendation System (Content-Based)

A content-based movie recommender built using the TMDB dataset and deployed with Streamlit.
Recommendations are generated using CountVectorizer + cosine similarity, trained offline and served via fast inference in production.

🚀 Features

Content-based recommendations

CountVectorizer for text representation

Cosine similarity for movie-to-movie matching

Offline training, online inference

Cloud-ready deployment (Streamlit)

🧠 Approach

Combine movie metadata (genres, keywords, overview, cast & crew)

Vectorize text using CountVectorizer

Compute cosine similarity

Save similarity matrix and metadata as .pkl

Load artifacts for real-time recommendations

📂 Project Structure
data/        # TMDB datasets
models/      # Serialized artifacts (.pkl)
src/         # Recommendation logic
training/    # Offline training pipeline
streamlit_app.py
requirements.txt

▶️ Run Locally
pip install -r requirements.txt
python -m streamlit run streamlit_app.py

🌐 Deployment

Streamlit entry file at repo root

No training in production

Compatible with Streamlit Cloud, Docker, AWS

👤 Author

Tejas Arondekar