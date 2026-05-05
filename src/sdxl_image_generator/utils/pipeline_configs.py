from dataclasses import dataclass
from sdxl_image_generator.utils.utils import BasePipelineConfig, LoraKey

@dataclass
class SDXLConfig(BasePipelineConfig):
    model_checkpoint: str
    scheduler: object | None = None
    init_loras: dict[str, LoraKey] | None = None