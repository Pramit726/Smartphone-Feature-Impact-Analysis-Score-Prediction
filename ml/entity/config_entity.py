from dataclasses import dataclass

from ml.constants.constants import *


@dataclass
class PredictionConfig:
    tracking_uri: str = MLFLOW_TRACKING_URI
    model_name: str = MODEL_NAME
    training_data_name: str = TRAINING_DATA_NAME
    test_input_data_name: str = TEST_X_DATA_NAME
    test_output_data_name: str = TEST_Y_DATA_NAME
