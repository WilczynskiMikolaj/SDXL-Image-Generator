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
from sdxl_image_generator.utils.utils import PACKAGE_ROOT, PipelineType, ModelDevice

class ModelLoaderBase(ABC):
    def __init__(self, available_models=None, available_loras=None, schedulers=None):
        self.loaded_model_name = None
        self.loaded_refiner_name = None
        self.active_loras = None
        self.active_scheduler = None

        self.pipe: Union[StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline] = None
        self.compel: CompelForSDXL = None
        self.refiner: StableDiffusionXLImg2ImgPipeline = None

        self.available_models = available_models or ["Default (stable-diffusion-xl-base-1.0)"]
        self.available_loras = available_loras or []
        self.available_schedulers = schedulers or {
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "ddim": DDIMScheduler,
            "heun": HeunDiscreteScheduler}
        
        self.models_directory: Path = PACKAGE_ROOT / "model_checkpoints"
        self.refiner_directory: Path = PACKAGE_ROOT / "refiners"
        self.loras_directory = PACKAGE_ROOT / "loras"

        self.refiner_on_gpu = False
        self.pipeline_on_gpu = False


    def load_loras(self, loras, adapter_weights=None):
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")

        if not loras:
            self.pipe.disable_lora()
            self.active_loras = []
            return
        
        if self.active_loras:
            self.pipe.disable_lora()

        if not isinstance(loras, list):
            raise TypeError("loras must be a list")

        for lora in loras:
            if lora not in self.available_loras:
                raise ValueError(f"LoRA {lora} not available")

        if adapter_weights is None:
            adapter_weights = [0.8] * len(loras)

        elif not isinstance(adapter_weights, list):
            raise TypeError("adapter_weights must be a list")

        elif len(adapter_weights) != len(loras):
            raise ValueError("adapter_weights length must match loras")

        for lora in loras:
            self.pipe.load_lora_weights(
                str(self.loras_directory),
                weight_name=lora,
                adapter_name=lora)

        self.pipe.set_adapters(loras, adapter_weights=adapter_weights)

        self.active_loras = loras

    def _initialize_pipeline(self, model_name:str, use_gpu:bool=True):
        if model_name != "Default (stable-diffusion-xl-base-1.0)" and model_name.endswith((".safetensors", ".ckpt")):
            self.pipe = StableDiffusionXLPipeline.from_single_file(self.models_directory / model_name, torch_dtype=torch.float16, use_safetensors=True)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True)

        if self.active_scheduler:
            self.change_scheduler(self.active_scheduler)

        if use_gpu:
            self._safe_load(PipelineType.TEXT2IMG, ModelDevice.GPU)
        else:
            self.pipeline_on_gpu = False

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    
    def load_model(self, model_name):
        if self.pipe and model_name == self.loaded_model_name:
            return

        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not in available models")

        if self.pipe:
            del self.pipe
            if self.pipeline_on_gpu:
                self.clear_cache()

        self._initialize_pipeline(model_name)

        self.loaded_model_name = model_name
    
    def load_model_selection(self, available_models: list, models_directory: Union[str, Path]) -> None:
        self.models_directory = Path(models_directory)
        self.available_models = available_models
    
    def load_loras_selection(self, available_loras: list, loras_directory: Union[str, Path]) -> None:
        self.loras_directory = Path(loras_directory)
        self.available_loras = available_loras

    def change_scheduler(self, name: str):
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")

        name = name.lower()
        
        if name not in self.available_schedulers:
            raise ValueError(f"Unknown scheduler '{name}'. " f"Available: {list(self.available_schedulers.keys())}")
        SchedulerClass = self.available_schedulers[name]
        self.pipe.scheduler = SchedulerClass.from_config(self.pipe.scheduler.config)
        self.active_scheduler = name

    def initialize_compel(self):
        if not self.pipe:
            raise RuntimeError("No model loaded")
        self.compel = CompelForSDXL(
            pipe=self.pipe,
            device=self.pipe.device)
        
    def clear_cache(self):
        torch.cuda.empty_cache()

    def load_refiner(self, refiner_model):
        if self.refiner and refiner_model == self.loaded_refiner_name:
            return

        if self.refiner:
            del self.refiner
            if self.refiner_on_gpu:
                self.clear_cache()
        
        self._initialize_refiner(refiner_model)
        self.loaded_refiner_name = refiner_model

    def _initialize_refiner(self, refiner_model, load_on_gpu=True):
        if refiner_model != "Default (stable-diffusion-base-refiner-1.0)" and refiner_model.endswith((".safetensors", ".ckpt")):
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(self.refiner_directory / refiner_model, torch_dtype=torch.float16, use_safetensors=True)
        else:
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True)

        if self.pipe:
            self.refiner.vae = self.pipe.vae
            self.refiner.text_encoder_2 = self.pipe.text_encoder_2
        
        try:
            self.refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        if load_on_gpu:
            self._safe_load(PipelineType.REFINER, ModelDevice.GPU)
        else:
            self.refiner_on_gpu = False


    def _safe_load(self, pipe_type: PipelineType, device: ModelDevice):
        if pipe_type == PipelineType.TEXT2IMG:
            if device == ModelDevice.GPU:
                if self.refiner and self.refiner_on_gpu:
                    self._load_on_cpu(PipelineType.REFINER)
                    torch.cuda.empty_cache()
                    self._load_on_gpu(PipelineType.TEXT2IMG)
                else:
                    self._load_on_gpu(PipelineType.TEXT2IMG)
            elif device == ModelDevice.CPU:
                self._load_on_cpu(PipelineType.TEXT2IMG)
        elif pipe_type == PipelineType.IMG2IMG:
            if device == ModelDevice.GPU:
                if self.refiner and self.refiner_on_gpu:
                    self._load_on_cpu(PipelineType.REFINER)
                    torch.cuda.empty_cache()
                    self._load_on_gpu(PipelineType.IMG2IMG)
                else:
                    self._load_on_gpu(PipelineType.IMG2IMG)
            elif device == ModelDevice.CPU:
                self._load_on_cpu(PipelineType.IMG2IMG)
        elif pipe_type == PipelineType.REFINER:
            if device == ModelDevice.GPU:
                if self.pipe and self.pipeline_on_gpu:
                    self._load_on_cpu(PipelineType.TEXT2IMG)
                    torch.cuda.empty_cache()
                    self._load_on_gpu(PipelineType.REFINER)
                else:
                    self._load_on_gpu(PipelineType.REFINER)
            elif device == ModelDevice.CPU:
                self._load_on_cpu(PipelineType.REFINER)

    def _load_on_gpu(self, pipe_type: PipelineType):
        match pipe_type:
            case PipelineType.TEXT2IMG | PipelineType.IMG2IMG:
                self.pipe.to("cuda")
                self.pipeline_on_gpu = True
            case PipelineType.REFINER:
                self.refiner.to("cuda")
                self.refiner_on_gpu = True

    def _load_on_cpu(self, pipe_type: PipelineType):
        match pipe_type:
            case PipelineType.TEXT2IMG | PipelineType.IMG2IMG:
                self.pipe.to("cpu")
                self.pipeline_on_gpu = False
            case PipelineType.REFINER:
                self.refiner.to("cpu")
                self.refiner_on_gpu = False

        torch.cuda.empty_cache()
        
    def decode_latents(self, latents):
        with torch.no_grad():
            images = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor
            ).sample

        images = images.detach()

        from diffusers.image_processor import VaeImageProcessor
        processor = VaeImageProcessor()

        return processor.postprocess(images, output_type="pil")
    
    @abstractmethod
    def generate_images(self):
        pass

    @abstractmethod
    def img2img(self):
        pass

    @abstractmethod
    def refine_image(self):
        pass