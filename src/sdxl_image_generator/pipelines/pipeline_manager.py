from pathlib import Path
from typing import Optional
from sdxl_image_generator.utils.utils import LoraKey, ModelDevice, default_schedulers, PipelineRequest
from sdxl_image_generator.pipelines.pipeline_cache import PipelineCache
from sdxl_image_generator.pipelines.pipeline_factory import PipelineFactory

class PipelineManager:
    def __init__(
        self,
        available_models: Optional[dict[str, Path]] = None,
        available_loras: Optional[dict[str, LoraKey]] = None,
        schedulers=None,
        max_pipelines_on_gpu=1,
        max_pipelines_on_cpu=1
    ):
        self.available_models: dict[str, Path] = available_models or {
            "default_sdxl": Path("stable-diffusion-xl-base-1.0")
        }

        self.available_loras: dict[str, LoraKey] = available_loras or {}

        self.available_schedulers: dict = schedulers or default_schedulers

        self.cache = PipelineCache(max_pipelines_on_gpu, max_pipelines_on_cpu)
        self.temp_img = []

    def run_pipeline(self, request: PipelineRequest):
        model_path = self._resolve_model_path(request.key.model_name)

        if hasattr(request.config, "model_checkpoint"):
            request.config.model_checkpoint = str(model_path)

        self._validate_loras(request.config)
        self._validate_scheduler(request.config)

        pipeline, pipeline_device = self.cache.get(request.key)

        if pipeline is None:
            self.cache.ensure_free_space(request.device)
            pipeline = PipelineFactory.create(
                request.key.model_type,
                request.key.pipeline_type,
                model_config=request.config,
                init_device=request.device
            )
            self.cache.add(request.key, pipeline, request.device)
            
        elif pipeline_device == "cpu" and request.device == ModelDevice.GPU:
            pipeline = self.cache.ensure_on_gpu(request.key, pipeline)
            
        elif pipeline_device == "gpu" and request.device == ModelDevice.GPU:
             self.cache.gpu_cache.move_to_end(request.key)

        images, latent = pipeline.run_pipeline(
            request.generation_config,
            request.config 
        )

        self.temp_img = latent
        return images

    def _resolve_model_path(self, model_name: Optional[str]) -> Path | str:
            self._validate_model_name(model_name)

            path = self.available_models[model_name]

            if isinstance(path, str) and "/" in path:
                return path

            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Checkpoint path does not exist: {path}"
                )

            return path

    def _validate_model_name(self, model_name: Optional[str]):
        if model_name is None:
            raise ValueError("Model name must be provided")

        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(self.available_models.keys())}"
            )

    def _validate_loras(self, config):
        if not hasattr(config, "init_loras") or config.init_loras is None:
            return

        for name, lora in config.init_loras.items():
            if name not in self.available_loras:
                raise ValueError(
                    f"Lora '{name}' not registered. "
                    f"Available: {list(self.available_loras.keys())}"
                )

            if not Path(lora.file_path).exists():
                raise FileNotFoundError(
                    f"Lora file not found: {lora.file_path}"
                )

    def _validate_scheduler(self, config):
        if not hasattr(config, "scheduler") or config.scheduler is None:
            return

        if isinstance(config.scheduler, str):
            if config.scheduler not in self.available_schedulers:
                raise ValueError(
                    f"Scheduler '{config.scheduler}' not available. "
                    f"Available: {list(self.available_schedulers.keys())}")
            config.scheduler = self.available_schedulers[config.scheduler]

    def set_available_models(self, models: dict[str, Path]):
        self.available_models = models

    def set_available_loras(self, loras: dict[str, LoraKey]):
        self.available_loras = loras

    def validate_model(self, model_name: str) -> bool:
        return model_name in self.available_models