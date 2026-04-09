import numpy as np
import pandas as pd
from fastapi import APIRouter, status
from sklearn import set_config

from .. import schemas
from ..repository import ratings

# Enable pandas output for the pipeline
set_config(transform_output="pandas")


router = APIRouter(prefix="/ratings", tags=["Rating"])


@router.post(
    "/predict",
    status_code=status.HTTP_200_OK,
    response_model=schemas.ShowRatings,
    summary="Predict Smartphone Rating",
    description="Predicts a smartphone rating based on provided specifications.",
)
async def predict_rating(request: schemas.SmartphoneSpecs):
    """
    Predicts a smartphone rating.
    """
    return await ratings.predict_rating(request)

@router.get(
    "/interpret",
    status_code=status.HTTP_200_OK,
    response_model=schemas.ShowInterpretation,
    summary="Interpret Model Predictions",
    description="Provides model interpretation, including SHAP summary and force plots, to understand feature importance and impact on predictions.",
)
async def interpret():
    """
    Provides model interpretation.
    """
    return await ratings.interpret()

@router.get(
    "/categories",
    status_code=status.HTTP_200_OK,
    response_model=schemas.ShowCatagories,
    summary="Get Categorical Column Categories",
    description="Retrieves the unique categories for each categorical column used in the model.",
)
async def get_categories():
    """
    Retrieves categories of categorical columns.
    """
    return await ratings.get_categories()


@router.get(
    "/model_metrics",
    status_code=status.HTTP_200_OK,
    response_model=schemas.ShowModelMetrics,
    summary="Get Model Performance Metrics",
    description="Retrieves model performance metrics, including Mean Absolute Error (MAE) and R-squared (R2) score.",
)
async def get_model_metrics():
    """
    Retrieves model metrics.
    """
    return await ratings.get_model_metrics()
