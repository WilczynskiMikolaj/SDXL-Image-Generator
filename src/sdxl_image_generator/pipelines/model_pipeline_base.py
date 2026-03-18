from sdxl_image_generator.utils.utils import PipelineType, ModelDevice, ModelType
from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, model_type: ModelType, pipeline_type:PipelineType, init_device: ModelDevice, alias:str):
        self.model_type:ModelType = model_type
        self.pipeline_type:PipelineType = pipeline_type
        self.device:ModelDevice = init_device
        self.alias:str = alias

        self.pipe = None

    def switch_device(self, device:ModelDevice):
        match device:
            case ModelDevice.CPU:
                self.pipe.to("cpu")
                self.device = ModelDevice.CPU
            case ModelDevice.GPU:
                self.pipe.to("cuda")
                self.device = ModelDevice.GPU

    @abstractmethod
    def initialize_pipeline(self):
        pass

    @abstractmethod
    def destroy_pipeline(self):
        pass