from pathlib import Path
from typing import Union, List
import tkinter as tk
from tkinter import filedialog


PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ICON_PATH = PACKAGE_ROOT / "assets" / "file_explorer_icon.png"

def get_directory(folder: Union[str, Path]) -> Path:
    path = PACKAGE_ROOT / folder
    return path

def get_all_directory_elements(folder_name: Union[str, Path], project_directory: bool) -> List[str]:
    if project_directory:
        directory = get_directory(folder_name)
        return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix == ".safetensors")
    else:
        folder_name = Path(folder_name)
        return sorted(p.name for p in folder_name.iterdir() if p.is_file() and p.suffix == ".safetensors")
    

def choose_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected