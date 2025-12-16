import joblib
from pathlib import Path

def save_model(model, model_name):
    model_dir = Path("D:/Code_ML/energy_consumption_forecasting/models") # defining path
    model_dir.mkdir(exist_ok=True) # creating directory if not exists

    model_path = model_dir/f'{model_name}.joblib' # curating path

    joblib.dump(model,model_path) # dumping model

    return model_path # returning model_path 

def load_model(model_name):

    # definign path to load model
    model_path = Path("models")/f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Ensure the model is trained and saved before deployment."
        )

    # using joblib to load model
    return joblib.load(model_path)
