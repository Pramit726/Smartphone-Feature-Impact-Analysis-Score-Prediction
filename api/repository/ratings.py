import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sklearn import set_config

from ml.logger.logger import logging
from ml.pipeline.prediction_pipeline import PredictionData, Predictor

from .. import schemas

# Enable pandas output for the pipeline
set_config(transform_output="pandas")

# Initialize predictor
predictor = Predictor()


async def predict_rating(request: schemas.SmartphoneSpecs):
    """
    Predicts smartphone rating based on provided specifications.
    Handles potential exceptions during prediction.
    """
    try:
        data = PredictionData(
            price=request.price,
            brand_name=request.brand_name,
            has_5g=request.has_5g,
            has_nfc=request.has_nfc,
            has_ir_blaster=request.has_ir_blaster,
            num_cores=request.num_cores,
            processor_speed=request.processor_speed,
            processor_brand=request.processor_brand,
            ram_capacity=request.ram_capacity,
            internal_memory=request.internal_memory,
            fast_charging=request.fast_charging,
            screen_size=request.screen_size,
            resolution=request.resolution,
            refresh_rate=request.refresh_rate,
            num_rear_cameras=request.num_rear_cameras,
            num_front_cameras=request.num_front_cameras,
            primary_camera_rear=request.primary_camera_rear,
            primary_camera_front=request.primary_camera_front,
            fast_charging_available=request.fast_charging_available,
            extended_memory_available=request.extended_memory_available,
            extended_upto=request.extended_upto,
        )
        data_df = data.get_data_as_df()

        prediction = predictor.predict(data_df)

        return {
            "rating": f"{prediction[0]}",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {e}",
        )


async def interpret():
    try:
        data_df = predictor.stored_dataframe
        summary_plot, force_plot = predictor.interpret(data_df)
        return {
            "summary_plot": summary_plot,
            "force_plot": force_plot,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during interpretation: {e}",
        )


async def get_categories():
    try:
        brand_categories = predictor.training_data["brand_name"].unique().tolist()
        processor_categories = [
            str(item)
            for item in predictor.training_data["processor_brand"].unique().tolist()
        ]
        resolution_categories = [
            str(item)
            for item in predictor.training_data["resolution"].unique().tolist()
        ]
        primary_camera_rear_categories = [
            str(item)
            for item in predictor.training_data["primary_camera_rear"].unique().tolist()
        ]
        return {
            "brand_categories": brand_categories,
            "processor_categories": processor_categories,
            "resolution_categories": resolution_categories,
            "primary_camera_rear_categories": primary_camera_rear_categories,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during categories retrieval: {e}",
        )


async def get_model_metrics():
    try:
        mae, r2_score = predictor.model_metrics()
        return {"mae": mae, "r2_score": r2_score}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during metrics retrieval: {e}",
        )
