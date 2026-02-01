from diffusers import StableDiffusionXLPipeline
import torch
from compel import CompelForSDXL
from sdxl_image_generator.sdxl_model_pipeline.model_configs import model_files, lora_files
import json
from sdxl_image_generator.utils.config_loader import load_from_json
from sdxl_image_generator.sdxl_model_pipeline.model_loader_base import ModelLoaderBase


class ModelLoaderCLI(ModelLoaderBase):
    def __init__(self, image_size=(1024, 1024), guidance_scale=4.123817, inference_steps=40, images_per_prompt=5, seed=None, guidance_rescale=0.0, adapter_weights=None,):
        super().__init__()
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
        self.images_per_prompt = images_per_prompt
        self.seed = seed
        self.guidance_rescale = guidance_rescale
        self.adapter_weights = adapter_weights or []

    def generate_images(self, prompt, negative_prompt):
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







class ModelLoaderTEMP:
    def __init__(self, model_name=None, loras=[], image_size=(1024, 1024), inference_steps=40, guidance_scale=4.123817, images_per_prompt=5, adapter_weights=None):
        self.model_name = model_name
        self.loras = loras or []
        self.image_size = image_size
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.images_per_prompt = images_per_prompt
        self.adapter_weights = adapter_weights

        self.pipe = self.initialize_pipeline()
        self.compel = self.build_compel()
    
    @classmethod
    def from_cli(cls, args):
        model_config = {}

        if args.json:
            model_config.update(load_from_json(args.json))

        if args.model is not None:
            model_config["model_name"] = args.model

        if args.loras is not None:
            model_config["loras"] = args.loras

        if args.adapter_weights is not None:
            model_config["adapter_weights"] = args.adapter_weights

        if args.loras and args.adapter_weights:
            if len(args.loras) != len(args.adapter_weights):
                raise ValueError(
                    "Number of adapter_weights must match number of loras "
                    f"({len(args.adapter_weights)} vs {len(args.loras)})"
                )

        if args.width is not None or args.height is not None:
            model_config["image_size"] = (
                args.width if args.width is not None else 1024,
                args.height if args.height is not None else 1024,
            )

        if args.guidance_scale is not None:
            model_config["guidance_scale"] = args.guidance_scale

        if args.inference_steps is not None:
            model_config["inference_steps"] = args.inference_steps

        if args.image_quantity is not None:
            model_config["images_per_prompt"] = args.image_quantity

        return cls(**model_config)

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
        model_source = ( model_files[self.model_name] if self.model_name else "stabilityai/stable-diffusion-xl-base-1.0")

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
