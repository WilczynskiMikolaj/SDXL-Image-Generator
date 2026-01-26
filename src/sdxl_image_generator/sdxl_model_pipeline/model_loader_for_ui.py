from sdxl_image_generator.sdxl_model_pipeline.model_loader_base import ModelLoaderBase
import torch
import json

class ModelLoaderUI(ModelLoaderBase):
    def apply_config(self, config: dict):
        if not isinstance(config, dict):
            raise TypeError("config must be a dict")

        if "width" in config:
            self.image_width = int(config["width"])

        if "height" in config:
            self.image_height = int(config["height"])

        if "inference_steps" in config:
            self.inference_steps = int(config["inference_steps"])

        if "guidance_scale" in config:
            self.guidance_scale = float(config["guidance_scale"])

        if "images_per_prompt" in config:
            self.images_per_prompt = int(config["images_per_prompt"])

        if "seed" in config:
            self.seed = int(config["seed"])

        if "guidance_rescale" in config:
            self.guidance_rescale = float(config["guidance_rescale"])

        if config.get("scheduler"):
            self.change_scheduler(config["scheduler"])

        if "loras" in config:
            self.load_loras(
                config["loras"],
                config.get("adapter_weights")
            )

    def to_dict(self):
        excluded = {"pipe", "compel", "available_models", "available_loras", "images_per_prompt"}
        result = {}

        for key, value in self.__dict__.items():
            if key in excluded:
                continue
            try:
                json.dumps(value)
                result[key] = value
            except TypeError:
                result[key] = str(value)

        return result

    def generate_images(self, prompt, negative_prompt, config=None):
        if config:
            self.apply_config(config)

        if not self.pipe or not self.compel:
            raise RuntimeError("Model or Compel not initialized")

        if self.seed is None or self.seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            self.seed = seed
        else:
            seed = self.seed

        conditioning = self.compel(prompt, negative_prompt=negative_prompt)

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.inference_mode():
            generated_images = self.pipe(prompt_embeds=conditioning.embeds, 
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                num_inference_steps=self.inference_steps, guidance_scale=self.guidance_scale, 
                width=self.image_width, height=self.image_height, num_images_per_prompt=self.images_per_prompt, 
                generator=generator, guidance_rescale=self.guidance_rescale)
            
        return generated_images

