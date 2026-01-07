from pathlib import Path
from typing import Union, List

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

def get_directory(folder: Union[str, Path]) -> Path:
    path = PACKAGE_ROOT / folder
    #path.mkdir(parents=True, exist_ok=True)
    return path

def get_all_directory_elements(folder_name: Union[str, Path]) -> List[str]:
    directory = get_directory(folder_name)
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix == ".safetensors"
    )