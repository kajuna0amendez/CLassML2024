"""
linear regression subpackage.

This subpackage provides the fundamental components and logic that power the application.
"""

from .tools import predict, mean_squared_error, loss_and_gradients, fit

__all__ = ["predict", "mean_squared_error", "loss_and_gradients", "fit"]
