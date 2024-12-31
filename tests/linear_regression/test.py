"""
The unit test for Linear Regression
"""
import pytest
import numpy as np
from backend.classification.linear_regression import predict, mean_squared_error, loss_and_gradients, fit


@pytest.fixture(scope='function')
def sample_data():
    """
    Sample data for the linear regression model
    """
    # Sample dataset
    x_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([2, 4, 6, 8, 10])
    return x_data, y_data
