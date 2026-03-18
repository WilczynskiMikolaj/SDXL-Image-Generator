import torch
from sdxl_image_generator.pipelines.sdxl_general_pipeline import SDXLImgGenPipelines
from sdxl_image_generator.utils.utils import ModelDevice, ModelType, PipelineType
from diffusers import StableDiffusionXLPipeline

class SDXLTxt2ImgPipeline(SDXLImgGenPipelines):
    def __init__(self, model_type: ModelType, pipeline_type:PipelineType, init_device: ModelDevice, alias:str, scheduler, model_checkpoint:str, init_loras:dict):
        super().__init__(self, model_type, pipeline_type, init_device, alias, scheduler, model_checkpoint, init_loras)
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        if self.pipe:
            self.destroy_pipeline()

        dtype = torch.float16 if self.device == ModelDevice.GPU else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_checkpoint, torch_dtype=dtype, use_safetensors=True)

        if self.scheduler:
            self.pipe.scheduler = self.scheduler

        if self.device == ModelDevice.GPU:
            self.load_on_gpu()

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        self.initialize_loras()
        self.initialize_compel()
    
    def generate_images(self, config):
        if not self.pipe or not self.compel:
            raise RuntimeError("Model or Compel not initialized")
        
        if config["seed"] is None or config["seed"] < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            seed = config["seed"]

        conditioning = self.compel(config["positive_prompt"], negative_prompt=config["negative_prompt"])
        device = "cuda" if self.device == ModelDevice.GPU else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        with torch.inference_mode():
            generated_images = self.pipe(prompt_embeds=conditioning.embeds, 
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                num_inference_steps=config["inference_steps"], guidance_scale=config["guidance_scale"], 
                width=config["image_width"], height=config["image_height"], num_images_per_prompt=config["images_per_prompt"], 
                generator=generator, guidance_rescale=config["guidance_rescale"], output_type="latent")
            
        latents = generated_images.images
        preview_images = self.decode_latents(latents)

        if self.device == ModelDevice.GPU:
            torch.cuda.empty_cache()
        
        return preview_images, latents


class SDXLImg2ImgPipeline(SDXLImgGenPipelines):
    def __init__(self, model_type, pipeline_type, init_device, alias):
        super().__init__(model_type, pipeline_type, init_device, alias)
        self.compel = None
