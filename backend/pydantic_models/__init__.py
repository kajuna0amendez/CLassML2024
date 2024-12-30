"""
pydasntic models for the backend
"""

from .models_linear_regression import LinearRegressionInput
from .common_tools import ForbidExtraFields

__all__ = ["LinearRegressionInput", "ForbidExtraFields"]
