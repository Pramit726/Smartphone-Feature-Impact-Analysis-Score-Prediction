import base64
import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import metrics, set_config
from sklearn.pipeline import Pipeline

from ml.exception.exception import SmartPhonesException
from ml.logger.logger import logging
from ml.utils.main_utils import encode_fig, make_columns_readable


class Model:
    def __init__(
        self,
        preprocessing_object: Pipeline,
        trained_model_object: object,
        training_data: pd.DataFrame,
        test_input_data: pd.DataFrame,
        test_output_data: pd.DataFrame,
    ):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model
        :param training_data: Input training data
        """
        set_config(transform_output="pandas")
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.training_data = training_data
        self.training_data_trans = self.preprocessing_object.transform(
            self.training_data
        )
        self.test_input_data = test_input_data
        self.test_output_data = test_output_data
        self.model_pipe = Pipeline(
            steps=[
                ("preprocessor", self.preprocessing_object),
                ("model", self.trained_model_object),
            ]
        )
        self.explainer = shap.TreeExplainer(self.trained_model_object)

    def predict(self, dataframe: pd.DataFrame):
        """
        Predict the output using the trained model
        :param data: Input data to be predicted
        """
        try:

            prediction = self.model_pipe.predict(dataframe)
            logging.info("Prediction successful from estimator")
            return prediction
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def model_metrics(self):
        try:
            set_config(transform_output="pandas")
            X_test_trans = self.preprocessing_object.transform(self.test_input_data)
            y_test = self.test_output_data
            y_test_trans = y_test.loc[X_test_trans.index]
            y_pred_test = self.model_pipe.predict(self.test_input_data)
            mae = metrics.mean_absolute_error(y_test_trans, y_pred_test)
            r2_score = metrics.r2_score(y_test_trans, y_pred_test)
            logging.info("Metrics calculated successfully")
            return mae, r2_score
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def shap_summary(self) -> plt.figure:
        """
        Generate SHAP summary plot and return figure.
        """
        try:

            # train_data_trans = self.preprocessing_object.transform(self.training_data)
            shap_values = self.explainer.shap_values(self.training_data_trans)
            # Create SHAP summary plot
            fig = plt.figure(figsize=(8, 5))
            shap.summary_plot(
                shap_values=shap_values,
                features=self.training_data_trans,
                feature_names=make_columns_readable(self.training_data_trans.columns),
                show=False,
            )
            # Convert figure to base64 string
            img_base64 = encode_fig(fig)

            logging.info("SHAP summary plot generated and encoded successfully")
            return img_base64
        except Exception as e:
            raise SmartPhonesException(e, sys)

    def shap_force(self, data: pd.DataFrame) -> str:
        """
        Generate SHAP force plot and return figure.
        :param data: Input data to be predicted
        """
        try:
            inv_std = self.preprocessing_object.named_steps["feature_encoding"][
                "standard_scaling"
            ]
            inv_min_max = self.preprocessing_object.named_steps["feature_encoding"][
                "minmax_scaling"
            ]
            inv_ord_en = self.preprocessing_object.named_steps["feature_encoding"][
                "ordinal_encoding"
            ]
            data_trans = self.preprocessing_object.transform(data)
            choosen_instance = (
                data_trans.copy()
                .pipe(
                    lambda df: df.assign(
                        **dict(
                            zip(
                                [
                                    col
                                    for col in df.columns
                                    if col.startswith("standard_scaling")
                                ],
                                inv_std.inverse_transform(
                                    df[
                                        [
                                            col
                                            for col in df.columns
                                            if col.startswith("standard_scaling")
                                        ]
                                    ]
                                ).T,
                            )
                        )
                    )
                )
                .pipe(
                    lambda df: df.assign(
                        **dict(
                            zip(
                                [
                                    col
                                    for col in df.columns
                                    if col.startswith("minmax_scaling")
                                ],
                                inv_min_max.inverse_transform(
                                    df[
                                        [
                                            col
                                            for col in df.columns
                                            if col.startswith("minmax_scaling")
                                        ]
                                    ]
                                ).T,
                            )
                        )
                    )
                )
                .pipe(
                    lambda df: df.assign(
                        **dict(
                            zip(
                                [
                                    col
                                    for col in df.columns
                                    if col.startswith("ordinal_encoding")
                                ],
                                inv_ord_en.inverse_transform(
                                    df[
                                        [
                                            col
                                            for col in df.columns
                                            if col.startswith("ordinal_encoding")
                                        ]
                                    ]
                                ).T,
                            )
                        )
                    )
                )
            )

            # Get SHAP values
            shap_values = self.explainer.shap_values(data_trans)
            # Extract base value (expected model output)
            base_value = self.explainer.expected_value
            # Generate SHAP force plot in HTML format
            shap_html = shap.plots.force(
                base_value=base_value,
                shap_values=shap_values[0],
                features=choosen_instance,
                feature_names=make_columns_readable(choosen_instance.columns),
                matplotlib=False,
            )
            # Get SHAP JavaScript
            shap_js = shap.getjs()

            # Combine JS with HTML output
            full_html = f"<head>{shap_js}</head><body>{shap_html.html()}</body>"
            logging.info("SHAP force plot generated successfully")
            return full_html
        except Exception as e:
            raise SmartPhonesException(e, sys)
