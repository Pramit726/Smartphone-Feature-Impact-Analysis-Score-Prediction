import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn import set_config

from ml.entity.estimator import Model
from ml.entity.transform import get_transformer
from ml.exception.exception import SmartPhonesException
from ml.logger.logger import logging

# Enable pandas output for the pipeline
set_config(transform_output="pandas")


class LoadObjects:
    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        training_data_name: str,
        test_input_data_name: str,
        test_output_data_name: str,
    ):
        self.tracking_uri: str = tracking_uri
        self.model_name: str = model_name
        self.training_data_name: str = training_data_name
        self.test_input_data_name: str = test_input_data_name
        self.test_output_data_name: str = test_output_data_name
        self.loaded_model: Model = None
        self.loaded_preprocesor: object = None
        self.loaded_training_data: pd.Dataframe = None

    def load_preprocessor(self):
        try:
            transformer = get_transformer()
            transformer.fit(self.load_training_data())
            self.loaded_preprocessor = transformer
            logging.info("Preprocessor loaded successfully")
            return self.loaded_preprocessor
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def load_model(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Load the model directly from MLflow
            model_uri = f"models:/{self.model_name}@lat"
            self.loaded_model = mlflow.sklearn.load_model(model_uri)
            logging.info("Model loaded successfully")
            return self.loaded_model
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def load_training_data(self):
        try:
            training_data_path = (
                Path(__file__).parent.parent.parent
                / "artifacts"
                / self.training_data_name
            )
            training_data = pd.read_csv(training_data_path)
            logging.info("Training data loaded successfully")
            return training_data
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def load_test_input_data(self):
        try:
            test_input_data_path = (
                Path(__file__).parent.parent.parent
                / "artifacts"
                / self.test_input_data_name
            )
            test_input_data = pd.read_csv(test_input_data_path)
            logging.info("Test input data loaded successfully")
            return test_input_data
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def load_test_output_data(self):
        try:
            test_output_data_path = (
                Path(__file__).parent.parent.parent
                / "artifacts"
                / self.test_output_data_name
            )
            test_output_data = pd.read_csv(test_output_data_path)
            logging.info("Test output data loaded successfully")
            return test_output_data
        except Exception as e:
            raise SmartPhonesException(e, sys)
