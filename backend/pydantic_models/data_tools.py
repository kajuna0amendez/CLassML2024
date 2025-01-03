"""
Build the Pydantic models for the FastAPI application.
"""
#pylint: disable=too-few-public-methods
from typing import List
from pydantic import Field, field_validator
from backend.pydantic_models import ForbidExtraFields


class FeatureRow(ForbidExtraFields):
    """
    Pydantic model for the x and y data samples
    """
    x_data: List[float] = Field(..., alias="xData")
    y_label: int = Field(..., alias="yLabel")

    @field_validator("x_data")
    @classmethod
    def check_xdata(cls, v):
        """
        Check that the 'xData' field is not an empty list
        """
        if not v:
            raise ValueError("The 'xData' field must not be an empty list")
        if not all(isinstance(x, float) for x in v):  # Ensure all elements are floats
            raise ValueError("All elements in 'total' must be floats.")
        return v

    @field_validator("y_label")
    @classmethod
    def check_ylabel(cls, v):
        """
        Check that the 'yLabel' field is not empty
        """
        if not v:
            raise ValueError("The 'yLabel' field must not be an empty")
        if not isinstance(v, int):
            raise ValueError("The 'yLabel' field must be an integer")
        return v


class DataInput(ForbidExtraFields):
    """
    Pydantic model for the input data to the FastAPI application
    """
    rows: List[FeatureRow]
