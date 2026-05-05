import torch
from sdxl_image_generator.utils.utils import BasePipelineConfig, GenerationConfig, PipelineType, ModelDevice, ModelType
from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, model_type: ModelType, pipeline_type:PipelineType, init_device: ModelDevice, model_config:BasePipelineConfig):
        self.model_type:ModelType = model_type
        self.pipeline_type:PipelineType = pipeline_type
        self.device:ModelDevice = init_device
        self.model_config = model_config

        self.pipe = None

    @abstractmethod
    def switch_device(self, device: ModelDevice):
        pass

    @abstractmethod
    def load_on_gpu(self):
        pass

    @abstractmethod
    def load_on_cpu(self):
        pass

    @abstractmethod
    def initialize_pipeline(self):
        pass

    @abstractmethod
    def destroy_pipeline(self):
        pass

    @abstractmethod
    def run_pipeline(self, gen_config: GenerationConfig, pipeline_config: BasePipelineConfig):
        pass