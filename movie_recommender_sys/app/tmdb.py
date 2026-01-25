import requests

TMDB_API_KEY = "a93c7c8dd53536ec8e0db47478a139ce"
BASE_URL = "https://api.themoviedb.org/3/movie"

def fetch_movie_details(movie_id: int):
    if TMDB_API_KEY is None:
        raise RuntimeError("TMDB_API_KEY is not set")

    url = f"{BASE_URL}/{movie_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    return {
        "title": data.get("title"),
        "overview": data.get("overview"),
        "poster_path": data.get("poster_path"),
        "rating": data.get("vote_average"),
        "release_date": data.get("release_date")
    }


if __name__ == "__main__":
    movie = fetch_movie_details(19995)  # Avatar
    print(movie)