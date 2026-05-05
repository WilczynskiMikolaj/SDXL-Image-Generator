from typing import Optional
from diffusers import SchedulerMixin
import torch
from sdxl_image_generator.pipelines.model_pipeline_base import BasePipeline
from sdxl_image_generator.utils.pipeline_configs import SDXLConfig
from sdxl_image_generator.utils.utils import ModelDevice, ModelType, PipelineType, LoraKey, BasePipelineConfig
from compel import CompelForSDXL

class SDXLImgGenPipelines(BasePipeline):
    def __init__(self, model_type: ModelType, pipeline_type:PipelineType, init_device: ModelDevice, model_config: SDXLConfig):
        super().__init__(model_type, pipeline_type, init_device, model_config)
        self.config: SDXLConfig = model_config

        self.loras: dict[str, LoraKey] = self.config.init_loras or {}
        self.scheduler: Optional[SchedulerMixin] = self.config.scheduler
        self.model_checkpoint: str = self.config.model_checkpoint
        
        self.compel:CompelForSDXL = None

    def update_scheduler(self, scheduler):
        if scheduler == self.scheduler:
            return
        self.scheduler = scheduler
        if self.pipe and scheduler is not None:
            self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)

    def initialize_compel(self):
        if not self.pipe:
            raise RuntimeError("No model loaded")
        self.compel = CompelForSDXL(
            pipe=self.pipe,
            device=self.pipe.device)

    def decode_latents(self, latents):
            with torch.no_grad():
                latents = latents.to(dtype=self.pipe.vae.dtype)
                
                images = self.pipe.vae.decode(
                    latents / self.pipe.vae.config.scaling_factor
                ).sample

            images = images.detach()

            from diffusers.image_processor import VaeImageProcessor
            processor = VaeImageProcessor()

            return processor.postprocess(images, output_type="pil")

    def destroy_pipeline(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
        if self.device == ModelDevice.GPU:
            torch.cuda.empty_cache()

    def update_lora(self, loras: dict[str, LoraKey] | None):
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")

        if not loras:
            self.pipe.disable_lora()
            self.loras = {}
            return

        adapter_names = []
        adapter_weights = []

        for name, lora in loras.items():
            safe_name = name.replace(".", "_").replace("-", "_")
            
            if name not in self.loras:
                self.pipe.load_lora_weights(
                    lora.file_path,
                    adapter_name=safe_name
                )
                
            adapter_names.append(safe_name)
            adapter_weights.append(lora.adapter_weight)

        self.pipe.set_adapters(
            adapter_names,
            adapter_weights=adapter_weights
        )

        self.loras = loras
                
    def initialize_loras(self):
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")

        if not self.loras:
            self.pipe.disable_lora()
            return

        adapter_names = []
        adapter_weights = []

        for name, lora in self.loras.items():
            safe_name = name.replace(".", "_").replace("-", "_")
            
            self.pipe.load_lora_weights(
                lora.file_path,
                adapter_name=safe_name
            )
            
            adapter_names.append(safe_name)
            adapter_weights.append(lora.adapter_weight)

        self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    def switch_device(self, target_device: ModelDevice):
        if self.pipe is None:
            return

        if self.device == target_device:
            return

        if target_device == ModelDevice.CPU:
            self.pipe.to("cpu")
            torch.cuda.empty_cache() 
        elif target_device == ModelDevice.GPU:
            self.pipe.to("cuda")

        self.device = target_device
        
        self.initialize_compel()

    def load_on_gpu(self):
        self.pipe.to("cuda")
        self.initialize_compel()

    def load_on_cpu(self):
        self.pipe.to("cpu")
        torch.cuda.empty_cache()
        self.initialize_compel()