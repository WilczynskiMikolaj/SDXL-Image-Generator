from diffusers import StableDiffusionXLPipeline
import torch
from compel import CompelForSDXL
from abc import ABC, abstractmethod

class ModelLoaderBase(ABC):
    def __init__(self, image_size=(1024, 1024), inference_steps=40, guidance_scale=4.123817, images_per_prompt=5, adapter_weights=None, available_models=None, available_loras=None):
        self.loaded_model_name = None
        self.active_loras = None
        self.image_size = image_size
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.images_per_prompt = images_per_prompt
        self.adapter_weights = adapter_weights or []

        self.pipe: StableDiffusionXLPipeline = None
        self.compel: CompelForSDXL = None

        self.available_models = available_models or {}
        self.available_loras = available_loras or {}


    def load_loras(self, loras, adapter_weights=None):
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")

        if not loras:
            self.pipe.disable_lora()
            self.active_loras = []
            self.adapter_weights = None
            return

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
            weight_file = self.available_loras[lora]

            self.pipe.load_lora_weights(
                "./loras",
                weight_name=weight_file,
                adapter_name=lora
            )

        self.pipe.set_adapters(loras, adapter_weights=adapter_weights)

        self.active_loras = loras
        self.adapter_weights = adapter_weights

    def _initialize_pipeline(self, model_name):
        model_source = self.available_models[model_name]
        if isinstance(model_source, str) and model_source.endswith((".safetensors", ".ckpt")):
            pipe = StableDiffusionXLPipeline.from_single_file(model_source, torch_dtype=torch.float16, use_safetensors=True)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_source, torch_dtype=torch.float16, use_safetensors=True)

        pipe = pipe.to("cuda")

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        self.pipe = pipe

    def load_model(self, model_name):
        if self.pipe and model_name == self.loaded_model_name:
            return

        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not in available models")

        if self.pipe:
            del self.pipe
            torch.cuda.empty_cache()

        self._initialize_pipeline(model_name)

        self.loaded_model_name = model_name


    def initialize_compel(self):
        if not self.pipe:
            raise RuntimeError("No model loaded")
        self.compel = CompelForSDXL(
            pipe=self.pipe,
            device=self.pipe.device)
        
    @abstractmethod
    def prompt_model(self):
        pass
