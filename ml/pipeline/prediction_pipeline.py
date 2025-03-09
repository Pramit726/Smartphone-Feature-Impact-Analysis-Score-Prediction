import sys

import numpy as np
import pandas as pd
from sklearn import set_config

from ml.entity.config_entity import PredictionConfig
from ml.entity.estimator import Model
from ml.entity.load_objects import LoadObjects
from ml.exception.exception import SmartPhonesException
from ml.logger.logger import logging

# Enable pandas output for the pipeline
set_config(transform_output="pandas")


class PredictionData:
    def __init__(
        self,
        price: int,
        brand_name: str,
        has_5g: bool,
        has_nfc: bool,
        has_ir_blaster: bool,
        num_cores: float,
        processor_speed: float,
        processor_brand: str,
        ram_capacity: float,
        internal_memory: float,
        fast_charging: float,
        screen_size: float,
        resolution: str,
        refresh_rate: int,
        num_rear_cameras: int,
        num_front_cameras: str,
        primary_camera_rear: str,
        primary_camera_front: float,
        fast_charging_available: int,
        extended_memory_available: int,
        extended_upto: float,
    ):
        """
        Prediction Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.price = (price,)
            self.processor_speed = (processor_speed,)
            self.screen_size = (screen_size,)
            self.brand_name = (brand_name,)
            self.processor_brand = (processor_brand,)
            self.extended_upto = (extended_upto,)
            self.has_5g = (has_5g,)
            self.has_nfc = (has_nfc,)
            self.has_ir_blaster = (has_ir_blaster,)
            self.num_cores = (num_cores,)
            self.ram_capacity = (ram_capacity,)
            self.internal_memory = (internal_memory,)
            self.fast_charging = (fast_charging,)
            self.resolution = (resolution,)
            self.refresh_rate = (refresh_rate,)
            self.num_rear_cameras = (num_rear_cameras,)
            self.num_front_cameras = (num_front_cameras,)
            self.primary_camera_rear = (primary_camera_rear,)
            self.primary_camera_front = (primary_camera_front,)
            self.fast_charging_available = (fast_charging_available,)
            self.extended_memory_available = extended_memory_available
            logging.info("Prediction data set")

        except Exception as e:
            raise SmartPhonesException(e, sys) from e

    def get_data_as_dict(self):
        """
        Get the prediction data as dictionary
        Output: dictionary of all features of the trained model for prediction
        """
        try:
            data_dict = {
                "price": self.price,
                "processor_speed": self.processor_speed,
                "screen_size": self.screen_size,
                "brand_name": self.brand_name,
                "processor_brand": self.processor_brand,
                "extended_upto": self.extended_upto,
                "has_5g": self.has_5g,
                "has_nfc": self.has_nfc,
                "has_ir_blaster": self.has_ir_blaster,
                "num_cores": self.num_cores,
                "ram_capacity": self.ram_capacity,
                "internal_memory": self.internal_memory,
                "fast_charging": self.fast_charging,
                "resolution": self.resolution,
                "refresh_rate": self.refresh_rate,
                "num_rear_cameras": self.num_rear_cameras,
                "num_front_cameras": self.num_front_cameras,
                "primary_camera_rear": self.primary_camera_rear,
                "primary_camera_front": self.primary_camera_front,
                "fast_charging_available": self.fast_charging_available,
                "extended_memory_available": self.extended_memory_available,
            }
            logging.info("Created prediction data dictionary")
            return data_dict
        except Exception as e:
            raise SmartPhonesException(e, sys) from e

    def get_data_as_df(self) -> pd.DataFrame:
        """
        Get the prediction data as dataframe
        Output: dataframe of all features of the trained model for prediction
        """
        try:
            data_dict = self.get_data_as_dict()
            data_df = pd.DataFrame(data_dict)
            logging.info("Created prediction dataframe")
            return data_df
        except Exception as e:
            raise SmartPhonesException(e, sys)


class Predictor:
    def __init__(self, prediction_pipeline_config=PredictionConfig()):
        """
        Predictor constructor
        Input: prediction configuration
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self.load_objects = LoadObjects(
                tracking_uri=self.prediction_pipeline_config.tracking_uri,
                model_name=self.prediction_pipeline_config.model_name,
                training_data_name=self.prediction_pipeline_config.training_data_name,
                test_input_data_name=self.prediction_pipeline_config.test_input_data_name,
                test_output_data_name=self.prediction_pipeline_config.test_output_data_name,
            )
            self.loaded_preprocessor = self.load_objects.load_preprocessor()
            self.loaded_model = self.load_objects.load_model()
            self.training_data = self.load_objects.load_training_data()
            self.test_input_data = self.load_objects.load_test_input_data()
            self.test_output_data = self.load_objects.load_test_output_data()
            self.model = Model(
                self.loaded_preprocessor,
                self.loaded_model,
                self.training_data,
                self.test_input_data,
                self.test_output_data,
            )

            logging.info("Prediction configuration set")
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> float:
        """
        Predict the output using the trained model
        Input: dataframe of all features of the trained model for prediction
        Output: predicted price
        """
        try:
            self.store_dataframe(dataframe)
            prediction = self.model.predict(dataframe)
            logging.info("Prediction successful from predictor")
            return prediction
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def store_dataframe(self, data_df: pd.DataFrame):
        try:
            self.stored_dataframe = data_df
            logging.info("Dataframe stored successfully")
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def model_metrics(self):
        try:
            mae, r2_score = self.model.model_metrics()
            logging.info("Model metrics successful from predictor")
            return mae, r2_score
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def interpret(self, data: pd.DataFrame):
        """
        Generate SHAP summary plot and return figure.
        Input: dataframe of all features of the trained model for prediction
        Output: SHAP summary plot
        """
        try:
            summary_plot = self.model.shap_summary()
            force_plot = self.model.shap_force(data)
            logging.info("Interpretation successful from predictor")
            return summary_plot, force_plot
        except Exception as e:
            raise SmartPhonesException(e, sys)
