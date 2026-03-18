import torch
from sdxl_image_generator.pipelines.model_pipeline_base import BasePipeline
from sdxl_image_generator.utils.utils import ModelDevice, ModelType, PipelineType, LoraKey
from compel import CompelForSDXL

class SDXLImgGenPipelines(BasePipeline):
    def __init__(self, model_type: ModelType, pipeline_type:PipelineType, init_device: ModelDevice, alias:str, scheduler, model_checkpoint:str, init_loras:dict):
        super().__init__(model_type, pipeline_type, init_device, alias)
        self.loras: dict[str, LoraKey] = init_loras or {}
        self.scheduler = scheduler
        self.model_checkpoint:str = model_checkpoint
        self.compel:CompelForSDXL = None

    def change_scheduler(self, scheduler):
        self.scheduler = scheduler
        if self.pipe:
            self.pipe.scheduler = scheduler

    def initialize_compel(self):
        if not self.pipe:
            raise RuntimeError("No model loaded")
        self.compel = CompelForSDXL(
            pipe=self.pipe,
            device=self.pipe.device)

    def decode_latents(self, latents):
        with torch.no_grad():
            images = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor
            ).sample

        images = images.detach()

        from diffusers.image_processor import VaeImageProcessor
        processor = VaeImageProcessor()

        return processor.postprocess(images, output_type="pil")
    
    def load_on_gpu(self):
        self.pipe.to("cuda")

    def load_on_cpu(self):
        self.pipe.to("cpu")
        torch.cuda.empty_cache()

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

        for name, lora in loras.items():
            if name not in self.loras:
                self.pipe.load_lora_weights(
                    lora.file_path,
                    adapter_name=name
                )

        self.pipe.set_adapters(
            list(loras.keys()),
            adapter_weights=[l.adapter_weight for l in loras.values()]
        )

        self.loras = loras
                
    def initialize_loras(self):
        if not self.pipe:
            raise RuntimeError("Pipeline not initialized")

        if not self.loras:
            self.pipe.disable_lora()
            return

        for name, lora in self.loras.items():
            self.pipe.load_lora_weights(
                lora.file_path,
                adapter_name=name
            )

        self.pipe.set_adapters(list(self.loras.keys()), adapter_weights=[l.adapter_weight for l in self.loras.values()])
