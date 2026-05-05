from collections import OrderedDict
from sdxl_image_generator.utils.utils import ModelDevice, PipelineKey
from sdxl_image_generator.pipelines.model_pipeline_base import BasePipeline

class PipelineCache:
    def __init__(self, max_gpu, max_cpu):
        self.gpu_cache: OrderedDict[PipelineKey, BasePipeline] = OrderedDict()
        self.cpu_cache: OrderedDict[PipelineKey, BasePipeline] = OrderedDict()

        self.max_gpu: int = max_gpu
        self.max_cpu: int = max_cpu

    def get(self, key):
        if key in self.cpu_cache:
            return self.cpu_cache[key], "cpu"
        
        if key in self.gpu_cache:
            return self.gpu_cache[key], "gpu"
        
        return None, None
    
    def ensure_free_space(self, device):
        cache = self.gpu_cache if device == ModelDevice.GPU else self.cpu_cache
        
        if device == ModelDevice.GPU and len(self.gpu_cache) >= self.max_gpu:
            self.evict(cache, ModelDevice.GPU)
            
        if device == ModelDevice.CPU and len(self.cpu_cache) >= self.max_cpu:
            self.evict(cache, ModelDevice.CPU)

    def add(self, key:PipelineKey, pipeline: BasePipeline, device: ModelDevice):
        cache = self.gpu_cache if device == ModelDevice.GPU else self.cpu_cache
        max_size = self.max_gpu if device == ModelDevice.GPU else self.max_cpu

        is_new = key not in cache

        # not sure what to do here now
        if is_new and len(cache) >= max_size:
            self.evict(cache, device)

        cache[key] = pipeline
        cache.move_to_end(key)

    def evict(self, cache, device: ModelDevice):
        old_key, old_pipeline = cache.popitem(last=False)

        if device == ModelDevice.GPU:
            if len(self.cpu_cache) >= self.max_cpu:
                self.evict(self.cpu_cache, ModelDevice.CPU)
            
            old_pipeline.switch_device(ModelDevice.CPU)
            self.cpu_cache[old_key] = old_pipeline
            
        elif device == ModelDevice.CPU:
            old_pipeline.destroy_pipeline()

    def ensure_on_gpu(self, key: PipelineKey, pipeline: BasePipeline) -> BasePipeline:
            if key in self.gpu_cache:
                self.gpu_cache.move_to_end(key)
                return pipeline

            if key in self.cpu_cache:
                del self.cpu_cache[key]

            if len(self.gpu_cache) >= self.max_gpu:
                self.evict(self.gpu_cache, ModelDevice.GPU)

            pipeline.switch_device(ModelDevice.GPU)

            self.gpu_cache[key] = pipeline
            self.gpu_cache.move_to_end(key)

            return pipeline