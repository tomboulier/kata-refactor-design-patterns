"""
Minimal functional pipeline DSL (Domain Specific Language) for image processing.

This module provides a very simple, elegant syntax to build pipelines
by composing transformation steps with the | operator.

Each step is a function wrapped into a Pipeline object via the @Pipeline.step decorator.

Examples
--------
>>> import numpy as np

Create a toy image:
>>> img = np.array([[0, 50], [100, 150]], dtype=np.uint8)

Build a simple pipeline:
>>> pipeline = normalize | to_float32
>>> out = pipeline(img)
>>> float(out.min()), float(out.max())
(0.0, 1.0)

Chaining multiple steps:
>>> pipeline = normalize | crop | to_float32
>>> out = pipeline(img)
>>> out.dtype == np.float32
True

Pipelines are immutable:
>>> p1 = normalize
>>> p2 = p1 | to_float32
>>> p1 is p2
False
"""

from __future__ import annotations
from typing import Callable, List
import numpy as np


Array = np.ndarray
Step = Callable[[Array], Array]


class Pipeline:
    """A minimalistic functional pipeline."""

    def __init__(self, steps: List[Step] | None = None):
        self.steps = steps or []

    def __or__(self, other: "Pipeline") -> "Pipeline":
        """Compose pipelines with the | operator (returns a new pipeline)."""
        return Pipeline(self.steps + other.steps)

    def __call__(self, data: Array) -> Array:
        """Run all steps on the data sequentially."""
        for step in self.steps:
            data = step(data)
        return data
    
    @classmethod
    def step(cls, fn: Step) -> Pipeline:
        """
        Decorator that turns a plain function into a Pipeline step.

        Examples
        --------
        >>> @Pipeline.step
        ... def double(x):
        ...     return x * 2
        >>> import numpy as np
        >>> p = double
        >>> p(np.array([1, 2, 3])).tolist()
        [2, 4, 6]
        """
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return Pipeline([wrapper])


# --------------------------------------------------------------------
# Steps definitions (each decorated with @Pipeline.step)
# --------------------------------------------------------------------

@Pipeline.step
def normalize(data: Array) -> Array:
    """Normalize array to [0, 1]."""
    min_val = data.min()
    max_val = data.max()
    if min_val == max_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


@Pipeline.step
def crop(data: Array) -> Array:
    """Crop the central square of the image."""
    h, w = data.shape
    size = min(h, w)
    sh = (h - size) // 2
    sw = (w - size) // 2
    return data[sh:sh + size, sw:sw + size]


@Pipeline.step
def to_float32(data: Array) -> Array:
    """Convert array to float32."""
    return data.astype(np.float32)


if __name__ == "__main__":
    # Execute the module’s doctests when run as a script.
    # This allows validating the examples in the docstrings using:
    #
    #     python image_pipeline_dsl.py
    #
    # Note: doctest compares string representations. When working with NumPy
    # arrays or scalar types, convert values to builtin Python types inside
    # examples if needed (e.g., `float(out.min())`).
    
    import doctest
    doctest.testmod()

