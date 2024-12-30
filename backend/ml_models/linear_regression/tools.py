"""
Basic Linear Regression Model
"""
import jax.numpy as jnp
from jax import grad
from jax.numpy import ndarray

def predict(x_data: ndarray, weights: ndarray, bias: ndarray)-> ndarray:
    """Liner Regression Prediction"""
    # Prediction: y = x_data*w + b
    return jnp.dot(weights, x_data) + bias

def mean_squared_error(x_data: ndarray, y: ndarray, weights: ndarray, bias: ndarray)-> ndarray:
    """Mean Squared Error Loss function"""
    # Mean Squared Error Loss function
    predictions = predict(x_data, weights, bias)
    return jnp.mean((predictions - y) ** 2)

def loss_and_gradients(x_data: ndarray, y: ndarray, weights: ndarray, bias: ndarray)-> ndarray:
    """gadient for  mean squared error"""
    assert x_data.shape[0] == y.shape[0], "x_data and y must have the same number of samples"
    weights = jnp.asarray(weights, dtype=jnp.float32)
    bias = jnp.asarray(bias, dtype=jnp.float32)
    x_data = jnp.asarray(x_data, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32) 
    # Compute the loss and its gradients w.r.t. weights and bias
    loss = mean_squared_error(x_data, y, weights, bias)
    grads = grad(mean_squared_error, argnums=[2, 3])(x_data, y, weights, bias)
    return loss, grads

def fit(x_data: ndarray, y: ndarray, weights: ndarray, bias: ndarray, learning_rate: float=0.001, epochs: int=100000)-> ndarray:
    """the fit method"""
    # Gradient descent to minimize loss
    for epoch in range(epochs):
        loss, grads = loss_and_gradients(x_data, y, weights, bias)
        weights -= learning_rate * grads[0]  # Update weights
        bias -= learning_rate * grads[1]     # Update bias
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    return weights, bias
