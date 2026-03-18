from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from compel import CompelForSDXL
from typing import Union
from abc import ABC, abstractmethod
from pathlib import Path
from diffusers import (
DPMSolverMultistepScheduler,
EulerAncestralDiscreteScheduler,
EulerDiscreteScheduler,
DDIMScheduler,
HeunDiscreteScheduler,
)
from sdxl_image_generator.utils.utils import PACKAGE_ROOT, PipelineType, ModelDevice, PipelineKey
from collections import OrderedDict
from sdxl_image_generator.pipelines.model_pipeline_base import BasePipeline

class PipelineManager(ABC):
    def __init__(self, available_models=None, available_loras=None, schedulers=None, max_pipelines_on_gpu=1, max_pipelines_on_cpu=1):
        self.active_pipelines = []

        self.available_models: list = available_models or ["Default (stable-diffusion-xl-base-1.0)"]
        self.available_loras: list = available_loras or []
        self.available_schedulers: dict = schedulers or {
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "heun": HeunDiscreteScheduler}
        
        self.models_directory: Path = PACKAGE_ROOT / "model_checkpoints"
        self.refiners_directory: Path = PACKAGE_ROOT / "refiners"
        self.loras_directory: Path = PACKAGE_ROOT / "loras"

        self.gpu_cache: OrderedDict[PipelineKey, BasePipeline] = OrderedDict()
        self.cpu_cache: OrderedDict[PipelineKey, BasePipeline] = OrderedDict()

        self.max_gpu_slots: int = max_pipelines_on_gpu
        self.max_cpu_slots: int = max_pipelines_on_cpu


    def load_loras(self, loras, adapter_weights=None):
        pass

    def _initialize_pipeline(self, model_name:str, use_gpu:bool=True):
        pass
    
    def load_model(self, model_name):
        pass
    
    def load_model_selection(self, available_models: list, models_directory: Union[str, Path]) -> None:
        pass
    
    def load_loras_selection(self, available_loras: list, loras_directory: Union[str, Path]) -> None:
        pass

    def change_scheduler(self, name: str):
        pass
        
    def clear_cache(self):
        pass

    def load_refiner(self, refiner_model):
        pass

    def _initialize_refiner(self, refiner_model, load_on_gpu=True):
        pass

    def _safe_load(self, pipe_type: PipelineType, device: ModelDevice):
        pass

    def _load_on_gpu(self, pipe_type: PipelineType):
        pass

    def _load_on_cpu(self, pipe_type: PipelineType):
        pass
    
    def load_upscaler(self):
        pass

    def _initialize_upscaler(self):
        pass
    
    @abstractmethod
    def generate_images(self):
        pass

    @abstractmethod
    def img2img(self):
        pass

    @abstractmethod
    def refine_image(self):
        pass