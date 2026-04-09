from unittest.mock import patch

import pandas as pd
import pytest

from ml.pipeline.prediction_pipeline import PredictionData, Predictor


@pytest.fixture
def sample_input_data():
    """Fixture to provide consistent test data."""
    return {
        "price": 15000,
        "brand_name": "Samsung",
        "has_5g": True,
        "has_nfc": True,
        "has_ir_blaster": False,
        "num_cores": 8.0,
        "processor_speed": 2.4,
        "processor_brand": "Snapdragon",
        "ram_capacity": 6.0,
        "internal_memory": 128.0,
        "fast_charging": 25.0,
        "screen_size": 6.7,
        "resolution": "1080x2400",
        "refresh_rate": 120,
        "num_rear_cameras": 3,
        "num_front_cameras": "1",
        "primary_camera_rear": "64",
        "primary_camera_front": 16.0,
        "fast_charging_available": 1,
        "extended_memory_available": 0,
        "extended_upto": 0.0,
    }


# --- Tests for PredictionData ---


def test_prediction_data_conversion(sample_input_data):
    """Verifies that input data is correctly converted to a DataFrame."""
    pred_data = PredictionData(**sample_input_data)
    df = pred_data.get_data_as_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 21)
    assert df["price"].iloc[0] == 15000
    assert df["brand_name"].iloc[0] == "Samsung"


# --- Tests for Predictor (Mocked) ---


@patch("ml.pipeline.prediction_pipeline.LoadObjects")
@patch("ml.pipeline.prediction_pipeline.Model")
def test_predictor_predict_logic(
    mock_model_class, mock_load_objects, sample_input_data
):
    """Tests if the predict method correctly interacts with the underlying model."""

    # Setup Mocks
    mock_model_instance = mock_model_class.return_value
    mock_model_instance.predict.return_value = 4.5  # Simulate a predicted rating

    # Initialize Predictorf
    predictor = Predictor()

    # Create input DF
    input_df = PredictionData(**sample_input_data).get_data_as_df()

    # Execute
    result = predictor.predict(input_df)

    # Assertions
    assert result == 4.5
    mock_model_instance.predict.assert_called_once_with(input_df)
    assert predictor.stored_dataframe is input_df


def test_predictor_metrics_logic():
    """Verifies that model_metrics returns expected values from the model class."""
    with patch("ml.pipeline.prediction_pipeline.LoadObjects"), patch(
        "ml.pipeline.prediction_pipeline.Model"
    ) as mock_model_class:

        mock_model_instance = mock_model_class.return_value
        mock_model_instance.model_metrics.return_value = (0.25, 0.88)  # MAE, R2

        predictor = Predictor()
        mae, r2 = predictor.model_metrics()

        assert mae == 0.25
        assert r2 == 0.88
