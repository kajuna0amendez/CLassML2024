"""
StandardScaler
"""
import jax.numpy as jnp
from jax.numpy import ndarray

def mean(data: ndarray, axis: int=0) -> ndarray:
    """
    Calculate the mean of the data
    """
    return jnp.mean(data, axis=axis)

def standard_deviation(data: ndarray, axis: int=0) -> ndarray:
    """
    Calculate the standard deviation of the data
    """
    return jnp.std(data, axis=axis)

def standard_scaler(data: ndarray) -> ndarray:
    """
    Standardize the data
    """
    return (data - mean(data)) / standard_deviation(data)
