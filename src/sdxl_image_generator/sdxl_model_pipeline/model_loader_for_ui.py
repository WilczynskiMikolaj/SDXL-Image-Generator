from sdxl_image_generator.sdxl_model_pipeline.model_loader_base import ModelLoaderBase
import torch

class ModelLoaderUI(ModelLoaderBase):
    def generate_images(self, config):
        if not self.pipe or not self.compel:
            raise RuntimeError("Model or Compel not initialized")

        if config["seed"] is None or config["seed"] < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            seed = config["seed"]

        conditioning = self.compel(config["prompt"], negative_prompt=config["negative_prompt"])

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.inference_mode():
            generated_images = self.pipe(prompt_embeds=conditioning.embeds, 
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                num_inference_steps=config["inference_steps"], guidance_scale=config["guidance_scale"], 
                width=config["image_width"], height=config["image_height"], num_images_per_prompt=config["images_per_prompt"], 
                generator=generator, guidance_rescale=config["guidance_rescale"])
            
        return generated_images
