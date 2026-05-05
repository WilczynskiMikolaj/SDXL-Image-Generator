from pathlib import Path
from typing import Union, List, Optional
import tkinter as tk
from tkinter import filedialog
import json
import re
import datetime
from enum import Enum
from dataclasses import dataclass
from diffusers import (
DPMSolverMultistepScheduler,
EulerAncestralDiscreteScheduler,
EulerDiscreteScheduler,
DDIMScheduler,
HeunDiscreteScheduler,
)
from diffusers import SchedulerMixin

PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SELECT_FOLDER_ICON_PATH = PACKAGE_ROOT / "assets" / "file_explorer_icon.png"
PROMPT_HISTORY_FILE = PACKAGE_ROOT / "logs" / "prompt_history.jsonl"
IMAGES_OUTPUT_FOLDER = PACKAGE_ROOT / "images"

class ModelType(Enum):
    SDXL = "sdxl"
    UPSCALER = "upscaler"

class PipelineType(Enum):
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    REFINER = "refiner"
    UPSCALER = "upscaler"

@dataclass(frozen=True)
class PipelineKey:
    model_type: ModelType
    pipeline_type: PipelineType
    model_name: Optional[str]

class ModelDevice(Enum):
    CPU = "cpu"
    GPU = "cuda"

def get_directory(folder: Union[str, Path]) -> Path:
    path = PACKAGE_ROOT / folder
    return path

@dataclass(frozen=True)
class LoraKey:
    file_path: str
    adapter_weight: float

@dataclass
class BasePipelineConfig:
    pass

@dataclass
class GenerationConfig:
    positive_prompt: str
    negative_prompt: str = ""

    inference_steps: int = 30
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0

    image_width: int = 1024
    image_height: int = 1024
    images_per_prompt: int = 1

    seed: Optional[int] = None

@dataclass
class PipelineRequest:
    key: PipelineKey
    config: BasePipelineConfig
    device: ModelDevice
    generation_config: GenerationConfig

default_schedulers =  {
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "heun": HeunDiscreteScheduler}


def get_all_directory_elements(folder_name: Union[str, Path], project_directory: bool) -> dict[str, Path]:
    if project_directory:
        directory = get_directory(folder_name)
    else:
        directory = Path(folder_name)
    
    return {p.name: p for p in directory.iterdir() if p.is_file() and p.suffix == ".safetensors"}
    

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
        cfg["image_width"],
        cfg["image_height"],
        cfg["inference_steps"],
        cfg["guidance_scale"],
        cfg["guidance_rescale"],
        cfg["images_per_prompt"],
        cfg["seed"]
    )

def save_all_images(generated_images, id, prompt, output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    safe_prompt = re.sub(r'[^\w\-_. ]', '_', prompt[:15])
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, image in enumerate(generated_images):
        filepath = output_folder / f"Prompt_{id}_{safe_prompt}_{i}_{current_time}.png"
        image.save(filepath)