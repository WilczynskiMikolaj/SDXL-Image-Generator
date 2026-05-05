import torch
from sdxl_image_generator.pipelines.sdxl_general_pipeline import SDXLConfig, SDXLImgGenPipelines
from sdxl_image_generator.utils.utils import ModelDevice, ModelType, PipelineType, GenerationConfig
from diffusers import StableDiffusionXLPipeline
from sdxl_image_generator.pipelines.pipeline_factory import PipelineFactory

@PipelineFactory.register(ModelType.SDXL, PipelineType.TEXT2IMG)
class SDXLTxt2ImgPipeline(SDXLImgGenPipelines):
    def __init__(self, model_type: ModelType, pipeline_type:PipelineType, init_device: ModelDevice,  model_config: SDXLConfig):
        super().__init__(model_type, pipeline_type, init_device, model_config)
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
            if self.pipe:
                self.destroy_pipeline()

            dtype = torch.float16 if self.device == ModelDevice.GPU else torch.float32
            
            checkpoint_path = str(self.model_checkpoint)

            if checkpoint_path.endswith(".safetensors") or checkpoint_path.endswith(".ckpt"):
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    checkpoint_path, 
                    torch_dtype=dtype, 
                    use_safetensors=True
                )
            else:
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    checkpoint_path, 
                    torch_dtype=dtype, 
                    use_safetensors=True
                )

            self.pipe.vae.to(dtype=torch.float32)

            if self.scheduler:
                self.pipe.scheduler = self.scheduler.from_config(self.pipe.scheduler.config)

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
    
    def run_pipeline(self, gen_config:GenerationConfig, pipeline_config: SDXLConfig):
        self.update_scheduler(pipeline_config.scheduler)
        self.update_lora(pipeline_config.init_loras)
        return self.generate_images(gen_config)

    def generate_images(self, config:GenerationConfig):
        if not self.pipe or not self.compel:
            raise RuntimeError("Model or Compel not initialized")
        
        if config.seed is None or config.seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            seed = config.seed

        conditioning = self.compel(config.positive_prompt, negative_prompt=config.negative_prompt)
        device = "cuda" if self.device == ModelDevice.GPU else "cpu"
        generator = torch.Generator(device).manual_seed(seed)

        with torch.inference_mode():
            generated_images = self.pipe(prompt_embeds=conditioning.embeds, 
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                num_inference_steps=config.inference_steps, guidance_scale=config.guidance_scale, 
                width=config.image_width, height=config.image_height, num_images_per_prompt=config.images_per_prompt, 
                generator=generator, guidance_rescale=config.guidance_rescale, output_type="latent")
            
        latents = generated_images.images
        preview_images = self.decode_latents(latents)

        if self.device == ModelDevice.GPU:
            torch.cuda.empty_cache()
        
        return preview_images, latents


class SDXLImg2ImgPipeline(SDXLImgGenPipelines):
    pass