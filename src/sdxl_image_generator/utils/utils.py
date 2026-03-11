from pathlib import Path
from typing import Union, List
import tkinter as tk
from tkinter import filedialog
import json


PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ICON_PATH = PACKAGE_ROOT / "assets" / "file_explorer_icon.png"
PROMPT_HISTORY_FILE = PACKAGE_ROOT / "prompt_history.jsonl"

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

def read_history_from_jsonl(json_file=None) -> list:
    if not json_file or not Path(json_file).exists():
        return []

    history = []

    with open(json_file, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))

    return history

def write_prompt_history(new_entry, json_file=None):
    if not json_file:
        return

    with open(json_file, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(new_entry) + "\n")

def apply_config(cfg):
    return (
        cfg["positive_prompt"],
        cfg["negative_prompt"],
        cfg["scheduler"],
        cfg["width"],
        cfg["height"],
        cfg["inference_steps"],
        cfg["guidance_scale"],
        cfg["guidance_rescale"],
        cfg["images_per_prompt"],
        cfg["seed"]
    )