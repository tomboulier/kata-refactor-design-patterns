# normalization.py

from __future__ import annotations
from typing import Dict, Callable, Literal
import numpy as np

NormalizationMethod = Callable[[np.ndarray], np.ndarray]

class NormalizationMethodsRegistry:
    """Registry for normalization methods."""
    
    registry: Dict[str, NormalizationMethod] = {}
    
    @classmethod
    def register(cls):
        """Decorator to register a normalization method."""
        def decorator(normalization_method: NormalizationMethod):
            method_name = normalization_method.__name__
            cls.registry[method_name] = normalization_method
            return normalization_method
        return decorator
    
    @classmethod
    def get(cls, method_name: str) -> NormalizationMethod:
        """Get a normalization method by name."""
        method = cls.registry.get(method_name)
        if not method:
            raise ValueError(f"Normalization method '{method_name}' not found.")
        return method


@NormalizationMethodsRegistry.register()
def minmax(data: np.ndarray) -> np.ndarray:
    """Normalize data using Min-Max normalization.

    This normalization scales the data to a fixed range [0, 1].
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)
    
@NormalizationMethodsRegistry.register()
def zscore(data: np.ndarray) -> np.ndarray:
    """Normalize data using Z-Score normalization.

    This normalization scales the data based on the mean and standard deviation.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data)
    return (data - mean) / std
    
@NormalizationMethodsRegistry.register()
def robust(data: np.ndarray) -> np.ndarray:
    """Normalize data using Robust normalization.

    This normalization scales the data based on the median and interquartile range (IQR).
    """
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    if iqr == 0:
        return np.zeros_like(data)
    return (data - median) / iqr
    
    

def normalize(
    data: np.ndarray,
    method: Literal["minmax", "zscore", "robust"]
) -> np.ndarray:
    """
    Normalize an array with the specified method.

    Parameters
    ----------
    data : np.ndarray
        Input vector.
    method : {"minmax", "zscore", "robust"}
        Normalization method.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    normalization_method = NormalizationMethodsRegistry.get(method)
    return normalization_method(data)

