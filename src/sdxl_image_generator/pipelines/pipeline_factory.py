from typing import Type
from sdxl_image_generator.pipelines.model_pipeline_base import BasePipeline
from sdxl_image_generator.utils.utils import ModelType, PipelineType

class PipelineFactory:
    _registry: dict[
        tuple[ModelType, PipelineType],
        Type[BasePipeline]
    ] = {}

    @classmethod
    def register(cls, model_type, pipeline_type):
        def decorator(pipeline_cls):
            key = (model_type, pipeline_type)

            if key in cls._registry:
                raise ValueError(f"Pipeline already registered for {key}")

            cls._registry[key] = pipeline_cls
            return pipeline_cls

        return decorator

    @classmethod
    def create(cls, model_type, pipeline_type, **kwargs):
        key = (model_type, pipeline_type)

        if key not in cls._registry:
            available = ", ".join(str(k) for k in cls._registry.keys())
            raise ValueError(
                f"No pipeline registered for {key}. Available: {available}"
            )

        return cls._registry[key](
            model_type=model_type, 
            pipeline_type=pipeline_type, 
            **kwargs
        )

    @classmethod
    def list_registered(cls):
        return list(cls._registry.keys())