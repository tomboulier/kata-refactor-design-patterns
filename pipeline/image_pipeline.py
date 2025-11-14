from __future__ import annotations
import numpy as np
from typing import Callable, Optional

Step = Callable[[np.ndarray], np.ndarray]

class Pipeline:
    """
    A simple pipeline to chain processing steps.
    
    Steps are functions that take and return a numpy ndarray.
    """
    _registry: dict[str, Step] = {}

    @classmethod
    def register_step(cls, name: str):
        def decorator(step: Step) -> Step:
            cls._registry[name] = step
            return step
        return decorator
    
    @classmethod
    def build(cls, **flags) -> Pipeline:
        pipeline = Pipeline()
        for name, enabled in flags.items():
            if name not in cls._registry:
                raise ValueError(f"Unknown pipeline step: {name}")
            if enabled:
                pipeline |= cls._registry[name]
        return pipeline
    
    def __init__(self, steps: Optional[list[Step]] = None):
        self.steps = steps or []

    def __or__(self, step: Step) -> Pipeline:
        return Pipeline(self.steps + [step])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for s in self.steps:
            data = s(data)
        return data
    
    
@Pipeline.register_step('normalize')
def normalization(data: np.ndarray) -> np.ndarray:
    """Normalize the input data to a range between 0 and 1."""
    min_val = data.min()
    max_val = data.max()
    if min_val == max_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)
    
@Pipeline.register_step('crop')
def crop_central_square(data: np.ndarray) -> np.ndarray:
    """Crop the central square of the input image."""
    h, w = data.shape
    size = min(h, w)
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return data[start_h:start_h+size, start_w:start_w+size]
    
@Pipeline.register_step('to_float32')
def convert_to_float32(data: np.ndarray) -> np.ndarray:
    """Convert the input data to float32 type."""
    return data.astype(np.float32)


def process_image(data: np.ndarray,
                  normalize: bool=True, 
                  crop: bool=True, 
                  to_float32: bool=True) -> np.ndarray:
    """
    Apply a configurable image processing pipeline.
    """
    pipeline = Pipeline.build(
        normalize=normalize, 
        crop=crop, 
        to_float32=to_float32
        )   
    return pipeline(data)
