"""
Example of linear regression using jax and pure functions
"""
import jax.numpy as jnp
from backend.ml_learn.regression.linear_regression import predict, fit
from config import eliminate_preallocate_memory

eliminate_preallocate_memory()

def train_prediction():
    """
    Train the model
    """
    x_data = jnp.array([-0.5, 0.5, -0.5, 0.5])
    y = jnp.array([1, 2, 1, 2])
    weights = jnp.array(0.0)
    bias = jnp.array(0.0)
    weights, bias = fit(x_data, y, weights, bias)
    print(weights, bias)
    return predict(x_data, weights, bias)

if __name__ == "__main__":
    print(train_prediction())
