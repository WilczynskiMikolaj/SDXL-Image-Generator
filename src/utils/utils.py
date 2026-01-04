from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_directory(folder):
    path = PROJECT_ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    return path