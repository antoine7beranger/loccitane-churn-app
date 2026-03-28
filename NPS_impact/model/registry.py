from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
print(f'mon base dir est{BASE_DIR}')
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
print(f'mon model dir est{MODEL_DIR}')

def save_pickle(obj, filename):
    filepath = MODEL_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    return filepath

def load_pickle(filename):
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


