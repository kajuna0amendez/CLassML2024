"""
Binary Linear Regression Model
"""
import jax.numpy as jnp
from jax.numpy import ndarray
from typing import Tuple

def fit(data: ndarray, labels: ndarray, weights: ndarray=None, bias: ndarray=None, solver: str="canonical") -> Tuple[ndarray, ndarray]:
    """
    Fit the linear regression model
    """
    if solver == "canonical":
        return __canonical_solution(data, labels)
    elif solver == "gradient_descent":
        if weights is None:
            pass # Initialize weights
        if bias is None:
            pass # Initialize bias
        return __gradient_descent(data, labels, weights, bias)
    elif solver == "nesterov_accelerated_gradient":
        return __nesterov_accelerated_gradient(data, labels, weights, bias)
    else:
        raise ValueError("Invalid solver")


def __canonical_solution(data: ndarray, labels: ndarray, weights: ndarray, bias: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Compute the canonical solution for the linear regression model
    """
    return jnp.linalg.pinv(data.T @ data) @ data.T @ labels, None


def __gradient_descent(data: ndarray, labels: ndarray, weights: ndarray, bias: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Compute the gradient descent solution for the linear regression model
    """
    return weights, bias


def __nesterov_accelerated_gradient(data: ndarray, labels: ndarray, weights: ndarray, bias: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Compute the Nesterov accelerated gradient solution for the linear regression model
    """
    return weights, bias