"""
Base classes for Pydantic models
"""
#pylint: disable=too-few-public-methods
from pydantic import BaseModel

class ForbidExtraFields(BaseModel):
    """
    Pydantic model to forbid extra fields in the request body
    It is minimal to be used as base class for other models
    """
    extras: str

    class Config:
        """
        Pydantic model configuration
        """
        extra = 'forbid'
