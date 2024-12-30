from fastapi import APIRouter
from pydantic_models import LinearRegressionInput
from typing import Dict

router = APIRouter()

@router.post("/linear_regression")
async def linear_regression(input: LinearRegressionInput)-> Dict:
    """
    Perform linear regression on the input data
    """
    return {"input": input, "output": "some output"}