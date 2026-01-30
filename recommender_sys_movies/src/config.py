from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "models"

MOVIES_PATH = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_PATH = DATA_DIR / "tmdb_5000_credits.csv"

ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
MAX_FEATURES = 5000