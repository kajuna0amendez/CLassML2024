"""
Services for linear regression
"""
from typing import Dict
from fastapi import APIRouter
from backend.pydantic_models import LinearRegressionInput


router = APIRouter()

@router.post("/linear_regression")
async def linear_regression(data: LinearRegressionInput)-> Dict:
    """
    Perform linear regression on the input data
    """
    return {"input": data, "output": "some output"}
