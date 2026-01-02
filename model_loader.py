from diffusers import StableDiffusionXLPipeline
import torch
from compel import CompelForSDXL
from model_configs import model_files, lora_files
import json

class ModelLoader:
    def __init__(self, model_name=None, loras=[], image_size=(1024, 1024), inference_steps=40, guidance_scale=4.123817, images_per_prompt=5, adapter_weights=None):
        self.model_name = model_name
        self.loras = loras
        self.image_size = image_size
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.images_per_prompt = images_per_prompt
        self.adapter_weights = adapter_weights
        self.pipe = self.initialize_pipeline()
        self.compel = self.build_compel()
    
    def load_loras(self):
        if not self.loras:
            self.pipe.disable_lora()
            return "No loras selected!"
        
        for lora in self.loras:
            self.pipe.load_lora_weights("./loras", weight_name=lora_files[lora], adapter_name=lora)

        adapter_weights = self.adapter_weights or self.set_default_adapter_weights()

        self.pipe.set_adapters(self.loras, adapter_weights=adapter_weights)
        return "Loras loaded!"
    
    def set_default_adapter_weights(self):
        return [0.8] * len(self.loras)
    
    def initialize_pipeline(self):
        model_source = (
                model_files[self.model_name]
            if self.model_name
            else "stabilityai/stable-diffusion-xl-base-1.0"
        )

        if self.model_name:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_source,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                use_safetensors=True
            )

        pipe = pipe.to("cuda")

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        return pipe

    def build_compel(self):
        return CompelForSDXL(
            pipe=self.pipe,
            device=self.pipe.device)
        
    def prompt_model(self):
        self.load_loras()
        with open("prompt.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        prompt = data.get("prompt", "")
        negative_prompt = data.get("negativePrompt", "")
        conditioning= self.compel(prompt, negative_prompt=negative_prompt)

        counter = 0
        with open("counter.txt", "r+") as f:
            counter = int(f.read()) 
            f.seek(0)
            f.write(f"{counter + self.images_per_prompt}")
            f.truncate()

        with torch.no_grad():
            generated_images = self.pipe(prompt_embeds=conditioning.embeds, 
                pooled_prompt_embeds=conditioning.pooled_embeds,
                negative_prompt_embeds=conditioning.negative_embeds,
                negative_pooled_prompt_embeds=conditioning.negative_pooled_embeds,
                num_inference_steps=self.inference_steps, guidance_scale=self.guidance_scale, 
                width=self.image_size[0], height=self.image_size[1], num_images_per_prompt=self.images_per_prompt)
        
        for i in range(self.images_per_prompt):
            generated_images.images[i].save(f"images/{counter + i}image.png")
