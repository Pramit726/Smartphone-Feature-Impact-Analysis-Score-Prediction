import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from ml.exception.exception import SmartPhonesException
from ml.logger.logger import logging

# Enable pandas output for the pipeline
set_config(transform_output="pandas")


standard_scale_cols = [
    "price",
    "processor_speed",
    "screen_size",
    "performance_score",
    "camera_quality",
    "storage_expandability",
]

minmax_scale_cols = ["connectivity_features"]

nominal_cols = ["brand_name", "processor_brand"]

ordinal_cols = [
    "extended_upto",
    "has_5g",
    "has_nfc",
    "has_ir_blaster",
    "num_cores",
    "ram_capacity",
    "internal_memory",
    "fast_charging",
    "resolution",
    "refresh_rate",
    "num_rear_cameras",
    "primary_camera_rear",
    "primary_camera_front",
    "fast_charging_available",
    "extended_memory_available",
]


ordinal_categories = [
    [
        "no expansion",
        "small",
        "medium",
        "large",
        "very large",
        "unknown",
    ],  # extended_upto
    [False, True],  # has_5g
    [False, True],  # has_nfc
    [False, True],  # has_ir_blaster
    [4.0, 6.0, 8.0],  # num_cores
    ["Low", "Standard", "High"],  # ram_capacity
    [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0],  # internal_memory
    ["no", "basic", "moderate", "fast", "ultra fast", "extreme fast"],  # fast_charging
    ["HD Ready", "Full HD", "Quad HD", "Ultra HD"],  # resolution
    [60, 90, 120, 144, 165],  # refresh_rate
    [1, 2, 3, 4],  # num_rear_cameras
    [
        "basic",
        "standard",
        "good",
        "high-Resolution",
        "ultra-high",
        "extreme",
    ],  # primary_camera_rear
    [
        "basic",
        "standard",
        "good",
        "high-resolution",
        "ultra high",
    ],  # primary_camera_front
    [0, 1],  # fast_charging_available
    [0, 1],  # extended_memory_available
]


class DropNA(BaseEstimator, TransformerMixin):
    """
    A transformer that drops rows with missing values (NaN) from a Pandas DataFrame.

    :param subset: List of columns to consider when dropping rows with NaN values.
                    If None, all columns are considered.
    :type subset: Optional[List[str]], optional

    Attributes
    ----------
    subset : Optional[List[str]]
        List of columns to consider when dropping rows with NaN values.
    """

    def __init__(self, subset: Optional[List[str]] = None):
        logging.info(f"Initializing DropNA transformer with subset: {subset}")
        self.subset = subset
        logging.info(f"DropNA transformer initialized.")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DropNA":
        """
        Fit the transformer. This method does not actually perform any computation.

        :param X: The input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: DropNA
        """
        logging.info("Fitting DropNA transformer.")
        logging.info("Fit completed. No computation performed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Drop rows with missing values (NaN) from the input DataFrame.

        :param X: The input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with rows containing NaN values in the specified subset of columns removed.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using DropNA.")
        logging.info(f"Original DataFrame shape: {X.shape}")
        X_transformed = X.dropna(subset=self.subset)
        logging.info(f"Transformed DataFrame shape: {X_transformed.shape}")
        logging.info("Transformation completed.")
        return X_transformed


class ProcessorBrandImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'processor_brand' values in a DataFrame based on the 'brand_name'.

    This transformer learns the most frequent 'processor_brand' for each 'brand_name'
    during the `fit` method and uses this information to impute missing values
    during the `transform` method. Handles cases where a brand has no associated
    processor brand in the training data, or where a specific brand ('ikall' or 'tesla')
    should always have an 'Unknown' processor.

    Attributes
    ----------
    brand_processor_mode : Dict[str, str]
        A dictionary mapping 'brand_name' to the most frequent 'processor_brand'.
        Stores the mode for each brand name learned during the fit step.
    """

    def __init__(self):
        logging.info("Initializing ProcessorBrandImputer.")
        self.brand_processor_mode: Optional[Dict[str, str]] = None
        logging.info("ProcessorBrandImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ProcessorBrandImputer":
        """
        Learns the most frequent 'processor_brand' for each 'brand_name'.

        Calculates the mode of 'processor_brand' for each 'brand_name' present
        in the input DataFrame `X` and stores it in `self.brand_processor_mode`.

        :param X: The input DataFrame containing 'brand_name' and 'processor_brand' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: ProcessorBrandImputer
        """
        logging.info("Fitting ProcessorBrandImputer.")
        self.brand_processor_mode = (
            X[X["processor_brand"].notnull()]
            .groupby("brand_name")["processor_brand"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
            .to_dict()
        )
        logging.info(f"Learned brand_processor_mode: {self.brand_processor_mode}")
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'processor_brand' values in the input DataFrame.

        Uses the learned `self.brand_processor_mode` to fill missing
        'processor_brand' values. If a 'brand_name' is not found in the
        learned mapping, 'Unknown' is used as the imputation value. Also handles
        special cases where 'ikall' and 'tesla' brands are always assigned
        the 'Unknown' processor.

        :param X: The input DataFrame containing 'brand_name' and 'processor_brand' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with missing 'processor_brand' values imputed.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using ProcessorBrandImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")

        def impute_processor(row):
            if row["brand_name"] in ["ikall", "tesla"]:
                logging.debug(f"Imputing 'Unknown' for brand: {row['brand_name']}")
                return "Unknown"
            elif pd.isnull(row["processor_brand"]):
                imputed_value = self.brand_processor_mode.get(
                    row["brand_name"], "Unknown"
                )
                logging.debug(
                    f"Imputing '{imputed_value}' for brand: {row['brand_name']}"
                )
                return imputed_value
            else:
                logging.debug(f"No imputation needed for brand: {row['brand_name']}")
                return row["processor_brand"]

        X["processor_brand"] = X.apply(impute_processor, axis=1)
        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class BatteryCapacityImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'battery_capacity' values with the median battery capacity
    of Apple devices.

    This transformer calculates the median 'battery_capacity' for devices where
    'brand_name' (case-insensitive) is 'apple' during the `fit` method and uses
    this median to fill missing 'battery_capacity' values during the `transform`
    method.

    Attributes
    ----------
    apple_median : Optional[float]
        The median battery capacity of Apple devices.
    """

    def __init__(self):
        logging.info("Initializing BatteryCapacityImputer.")
        self.apple_median: Optional[float] = None
        logging.info("BatteryCapacityImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BatteryCapacityImputer":
        """
        Calculates the median battery capacity for Apple devices.

        :param X: The input DataFrame containing 'brand_name' and 'battery_capacity' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: BatteryCapacityImputer
        """
        logging.info("Fitting BatteryCapacityImputer.")
        apple_data = X.loc[X["brand_name"].str.lower() == "apple", "battery_capacity"]
        self.apple_median = apple_data.median()
        logging.info(f"Calculated Apple median battery capacity: {self.apple_median}")
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'battery_capacity' values with the calculated median.

        :param X: The input DataFrame containing 'battery_capacity' column.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with missing 'battery_capacity' values imputed.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using BatteryCapacityImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")
        X["battery_capacity"] = X["battery_capacity"].fillna(self.apple_median)
        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class ExtendedUptoImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'extended_upto' values with the median 'extended_upto' value
    for devices where 'extended_memory_available' is 1.

    Attributes
    ----------
    extended_memory_median : Optional[float]
        The median 'extended_upto' value for devices with extended memory available.
    """

    def __init__(self):
        logging.info("Initializing ExtendedUptoImputer.")
        self.extended_memory_median: Optional[float] = None
        logging.info("ExtendedUptoImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ExtendedUptoImputer":
        """
        Calculates the median 'extended_upto' value for devices with extended memory.

        :param X: The input DataFrame containing 'extended_memory_available' and 'extended_upto' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: ExtendedUptoImputer
        """
        logging.info("Fitting ExtendedUptoImputer.")
        extended_memory_data = X.loc[
            X["extended_memory_available"] == 1, "extended_upto"
        ]
        self.extended_memory_median = extended_memory_data.median()
        logging.info(
            f"Calculated extended memory median: {self.extended_memory_median}"
        )
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'extended_upto' values.

        :param X: The input DataFrame containing 'extended_upto' column.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with missing 'extended_upto' values imputed.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using ExtendedUptoImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")

        def impute_extended_upto(row):
            if pd.isnull(row["extended_upto"]):
                logging.debug(
                    f"Imputing {self.extended_memory_median} for extended_upto."
                )
                return self.extended_memory_median
            else:
                logging.debug(f"extended_upto value present, no imputation needed.")
                return row["extended_upto"]

        X["extended_upto"] = X.apply(impute_extended_upto, axis=1)

        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class FastChargingImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'fast_charging' values to 0 when 'fast_charging_available' is 0.
    """

    def __init__(self):
        logging.info("Initializing FastChargingImputer.")
        logging.info("FastChargingImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FastChargingImputer":
        """
        Fit the transformer. This method does not actually perform any computation.

        :param X: The input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: FastChargingImputer
        """
        logging.info("Fitting FastChargingImputer. No computation performed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes 'fast_charging' values.

        Sets 'fast_charging' to 0 if 'fast_charging_available' is 0 and 'fast_charging' is NaN.

        :param X: The input DataFrame containing 'fast_charging' and 'fast_charging_available' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with imputed 'fast_charging' values.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using FastChargingImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")

        def impute_fast_charging(row):
            if row["fast_charging_available"] == 0 and pd.isnull(row["fast_charging"]):
                logging.debug("Imputing 0 for fast_charging.")
                return 0
            else:
                logging.debug(
                    "fast_charging value present or fast_charging_available is 1, no imputation needed."
                )
                return row["fast_charging"]

        X["fast_charging"] = X.apply(impute_fast_charging, axis=1)

        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class ConsistentBrandsImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'fast_charging' values for specific brands with their medians.

    This transformer calculates the median 'fast_charging' value for a predefined
    list of brands ('lg', 'oneplus', 'xiaomi', 'vivo') during the `fit` method.
    It then uses these medians to impute missing 'fast_charging' values for the
    respective brands during the `transform` method.

    Attributes
    ----------
    brand_medians : Dict[str, Optional[float]]
        A dictionary storing the median 'fast_charging' value for each
        consistent brand.
    """

    def __init__(self):
        logging.info("Initializing ConsistentBrandsImputer.")
        self.brand_medians: Dict[str, Optional[float]] = {}
        logging.info("ConsistentBrandsImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ConsistentBrandsImputer":
        """
        Calculates the median 'fast_charging' for consistent brands.

        :param X: The input DataFrame containing 'brand_name' and 'fast_charging' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: ConsistentBrandsImputer
        """
        logging.info("Fitting ConsistentBrandsImputer.")
        consistent_brands = ["lg", "oneplus", "xiaomi", "vivo"]
        for brand in consistent_brands:
            logging.debug(f"Calculating median for brand: {brand}")
            median_value = X.loc[
                X["brand_name"].str.lower() == brand, "fast_charging"
            ].median()
            self.brand_medians[brand] = median_value
            logging.debug(f"Median for brand {brand}: {median_value}")
        logging.info(f"Calculated brand medians: {self.brand_medians}")
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'fast_charging' values for consistent brands.

        :param X: The input DataFrame containing 'brand_name' and 'fast_charging' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with imputed 'fast_charging' values for consistent brands.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using ConsistentBrandsImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")
        for brand, median_value in self.brand_medians.items():
            logging.debug(f"Imputing median for brand: {brand}, median: {median_value}")
            X.loc[
                (X["brand_name"].str.lower() == brand) & (X["fast_charging"].isnull()),
                "fast_charging",
            ] = median_value
            logging.debug(f"Imputed values for brand: {brand}")
        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class HigherMissingBrandsImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'fast_charging' values for specific brands with their medians.

    This transformer calculates the median 'fast_charging' value for a predefined
    list of brands ('samsung', 'apple') during the `fit` method. It then uses
    these medians to impute missing 'fast_charging' values for the respective
    brands during the `transform` method.

    Attributes
    ----------
    brand_medians : Dict[str, Optional[float]]
        A dictionary storing the median 'fast_charging' value for each
        specified brand.
    """

    def __init__(self):
        logging.info("Initializing HigherMissingBrandsImputer.")
        self.brand_medians: Dict[str, Optional[float]] = {}
        logging.info("HigherMissingBrandsImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "HigherMissingBrandsImputer":
        """
        Calculates the median 'fast_charging' for specified brands.

        :param X: The input DataFrame containing 'brand_name' and 'fast_charging' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: HigherMissingBrandsImputer
        """
        logging.info("Fitting HigherMissingBrandsImputer.")
        higher_missing_brands = ["samsung", "apple"]
        for brand in higher_missing_brands:
            logging.debug(f"Calculating median for brand: {brand}")
            median_value = X.loc[
                X["brand_name"].str.lower() == brand, "fast_charging"
            ].median()
            self.brand_medians[brand] = median_value
            logging.debug(f"Median for brand {brand}: {median_value}")
        logging.info(f"Calculated brand medians: {self.brand_medians}")
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'fast_charging' values for specified brands.

        :param X: The input DataFrame containing 'brand_name' and 'fast_charging' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with imputed 'fast_charging' values for specified brands.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using HigherMissingBrandsImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")
        for brand, median_value in self.brand_medians.items():
            logging.debug(f"Imputing median for brand: {brand}, median: {median_value}")
            X.loc[
                (X["brand_name"].str.lower() == brand) & (X["fast_charging"].isnull()),
                "fast_charging",
            ] = median_value
            logging.debug(f"Imputed values for brand: {brand}")
        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class LowerMissingBrandsImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'fast_charging' values for specific brands with their medians.

    This transformer calculates the median 'fast_charging' value for a predefined
    list of brands ('sony', 'xiaomi', 'oppo', 'motorola', 'google') during the
    `fit` method. It then uses these medians to impute missing 'fast_charging'
    values for the respective brands during the `transform` method.

    Attributes
    ----------
    brand_medians : Dict[str, Optional[float]]
        A dictionary storing the median 'fast_charging' value for each
        specified brand.
    """

    def __init__(self):
        logging.info("Initializing LowerMissingBrandsImputer.")
        self.brand_medians: Dict[str, Optional[float]] = {}
        logging.info("LowerMissingBrandsImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "LowerMissingBrandsImputer":
        """
        Calculates the median 'fast_charging' for specified brands.

        :param X: The input DataFrame containing 'brand_name' and 'fast_charging' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: LowerMissingBrandsImputer
        """
        logging.info("Fitting LowerMissingBrandsImputer.")
        lower_missing_brands = ["sony", "xiaomi", "oppo", "motorola", "google"]
        for brand in lower_missing_brands:
            logging.debug(f"Calculating median for brand: {brand}")
            median_value = X.loc[
                X["brand_name"].str.lower() == brand, "fast_charging"
            ].median()
            self.brand_medians[brand] = median_value
            logging.debug(f"Median for brand {brand}: {median_value}")
        logging.info(f"Calculated brand medians: {self.brand_medians}")
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'fast_charging' values for specified brands.

        :param X: The input DataFrame containing 'brand_name' and 'fast_charging' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with imputed 'fast_charging' values for specified brands.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using LowerMissingBrandsImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")
        for brand, median_value in self.brand_medians.items():
            logging.debug(f"Imputing median for brand: {brand}, median: {median_value}")
            X.loc[
                (X["brand_name"].str.lower() == brand) & (X["fast_charging"].isnull()),
                "fast_charging",
            ] = median_value
            logging.debug(f"Imputed values for brand: {brand}")
        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


class SingleRecordBrandsRemover(BaseEstimator, TransformerMixin):
    """
    Removes records associated with specific brands from a DataFrame.

    This transformer removes rows where the 'brand_name' (case-insensitive) is
    present in a predefined or user-provided list of brands.

    Attributes
    ----------
    brands_to_remove : List[str], default=['blu', 'leitz', 'sharp']
        A list of brand names to remove from the DataFrame.
    """

    def __init__(self, brands_to_remove: Optional[List[str]] = None):
        logging.info("Initializing SingleRecordBrandsRemover.")
        self.brands_to_remove = brands_to_remove or [
            "blu",
            "leitz",
            "sharp",
        ]  # Default brands
        logging.info(f"Brands to remove: {self.brands_to_remove}")
        logging.info("SingleRecordBrandsRemover initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "SingleRecordBrandsRemover":
        """
        Fit the transformer. This method does not actually perform any computation.

        :param X: The input DataFrame.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: SingleRecordBrandsRemover
        """
        logging.info("Fitting SingleRecordBrandsRemover. No computation performed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Removes records associated with the specified brands.

        :param X: The input DataFrame containing the 'brand_name' column.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with records associated with the specified brands removed.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using SingleRecordBrandsRemover.")
        logging.info(f"Original DataFrame shape: {X.shape}")
        X_transformed = X[~X["brand_name"].str.lower().isin(self.brands_to_remove)]
        logging.info(f"Transformed DataFrame shape: {X_transformed.shape}")
        logging.info("Transformation completed.")
        return X_transformed


class OSByBrandImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing 'os' values based on the mode 'os' for each 'brand_name'.

    This transformer calculates the mode of the 'os' column for each
    'brand_name' during the `fit` method and uses these modes to impute
    missing 'os' values during the `transform` method.

    Attributes
    ----------
    os_mode : Dict[str, Optional[str]]
        A dictionary mapping each 'brand_name' to its most frequent 'os' value.
        Can be None if fit has not been called.
    """

    def __init__(self):
        logging.info("Initializing OSByBrandImputer.")
        self.os_mode: Optional[Dict[str, Optional[str]]] = None
        logging.info("OSByBrandImputer initialized.")

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "OSByBrandImputer":  # String literal
        """
        Calculates the mode 'os' for each 'brand_name'.

        :param X: The input DataFrame containing 'brand_name' and 'os' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: Returns self.
        :rtype: OSByBrandImputer
        """
        logging.info("Fitting OSByBrandImputer.")
        self.os_mode = (
            X.groupby("brand_name")["os"]
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .to_dict()
        )
        logging.info(f"Calculated os_mode: {self.os_mode}")
        logging.info("Fit completed.")
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Imputes missing 'os' values based on the calculated modes.

        :param X: The input DataFrame containing 'brand_name' and 'os' columns.
        :type X: pd.DataFrame
        :param y: Ignored. Present here for compatibility with the scikit-learn API.
        :type y: Optional[pd.Series], optional
        :return: The DataFrame with imputed 'os' values.
        :rtype: pd.DataFrame
        """
        logging.info("Transforming DataFrame using OSByBrandImputer.")
        logging.info(f"Original DataFrame shape: {X.shape}")

        def impute_os(row):
            if pd.isnull(row["os"]):
                imputed_value = self.os_mode.get(row["brand_name"])
                logging.debug(
                    f"Imputing '{imputed_value}' for brand: {row['brand_name']}"
                )
                return imputed_value
            else:
                logging.debug(f"No imputation needed for brand: {row['brand_name']}")
                return row["os"]

        X["os"] = X.apply(impute_os, axis=1)

        logging.info(f"Transformed DataFrame shape: {X.shape}")
        logging.info("Transformation completed.")
        return X


def compute_performance_score(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a performance score based on processor speed, number of cores, and RAM capacity.

    :param X: The input DataFrame containing 'processor_speed', 'num_cores', and 'ram_capacity' columns.
    :type X: pd.DataFrame
    :return: The DataFrame with the added 'performance_score' column.
    :rtype: pd.DataFrame
    """
    logging.info("Computing performance score.")
    logging.info(f"Original DataFrame shape: {X.shape}")
    X = X.copy()
    try:
        X["performance_score"] = (
            X["processor_speed"]
            * X["num_cores"].astype(float)
            * X["ram_capacity"].astype(float)
        )
        logging.info("Performance score computed successfully.")
    except Exception as e:
        logging.error(f"Error computing performance score: {e}")
        raise  # re-raise exception so it is not hidden.

    logging.info(f"Transformed DataFrame shape: {X.shape}")
    return X


def compute_battery_efficiency(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes battery efficiency based on battery capacity and screen size.

    :param X: The input DataFrame containing 'battery_capacity' and 'screen_size' columns.
    :type X: pd.DataFrame
    :return: The DataFrame with the added 'battery_efficiency' column.
    :rtype: pd.DataFrame
    """
    logging.info("Computing battery efficiency.")
    logging.info(f"Original DataFrame shape: {X.shape}")
    X = X.copy()
    try:
        X["battery_efficiency"] = X["battery_capacity"] / X["screen_size"]
        logging.info("Battery efficiency computed successfully.")
    except Exception as e:
        logging.error(f"Error computing battery efficiency: {e}")
        raise  # re-raise exception so it is not hidden.

    logging.info(f"Transformed DataFrame shape: {X.shape}")
    return X


def compute_camera_quality(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes camera quality based on rear and front camera megapixels.

    :param X: The input DataFrame containing 'primary_camera_rear' and 'primary_camera_front' columns.
    :type X: pd.DataFrame
    :return: The DataFrame with the added 'camera_quality' column.
    :rtype: pd.DataFrame
    """
    logging.info("Computing camera quality.")
    logging.info(f"Original DataFrame shape: {X.shape}")
    X = X.copy()
    try:
        X["camera_quality"] = X["primary_camera_rear"].astype(float) + X[
            "primary_camera_front"
        ].astype(float)
        logging.info("Camera quality computed successfully.")
    except Exception as e:
        logging.error(f"Error computing camera quality: {e}")
        raise  # re-raise exception so it is not hidden.

    logging.info(f"Transformed DataFrame shape: {X.shape}")
    return X


def compute_connectivity_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a connectivity features score based on 5g, NFC, and IR blaster availability.

    :param X: The input DataFrame containing 'has_5g', 'has_nfc', and 'has_ir_blaster' columns.
    :type X: pd.DataFrame
    :return: The DataFrame with the added 'connectivity_features' column.
    :rtype: pd.DataFrame
    """
    logging.info("Computing connectivity features.")
    logging.info(f"Original DataFrame shape: {X.shape}")
    X = X.copy()
    try:
        X["connectivity_features"] = X[["has_5g", "has_nfc", "has_ir_blaster"]].sum(
            axis=1
        )
        logging.info("Connectivity features computed successfully.")
    except Exception as e:
        logging.error(f"Error computing connectivity features: {e}")
        raise  # re-raise exception so it is not hidden.

    logging.info(f"Transformed DataFrame shape: {X.shape}")
    return X


def compute_storage_expandability(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes storage expandability based on maximum expandable memory and availability.

    :param X: The input DataFrame containing 'extended_upto' and 'extended_memory_available' columns.
    :type X: pd.DataFrame
    :return: The DataFrame with the added 'storage_expandability' column.
    :rtype: pd.DataFrame
    """
    logging.info("Computing storage expandability.")
    logging.info(f"Original DataFrame shape: {X.shape}")
    X = X.copy()
    try:
        X["storage_expandability"] = X["extended_upto"].astype(float) * X[
            "extended_memory_available"
        ].astype(int)
        logging.info("Storage expandability computed successfully.")
    except Exception as e:
        logging.error(f"Error computing storage expandability: {e}")
        raise  # re-raise exception so it is not hidden.

    logging.info(f"Transformed DataFrame shape: {X.shape}")
    return X


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical and numerical features into discrete categories.

    This function performs binning and grouping operations on several features
    to convert them into categorical representations. It handles 'resolution',
    'brand_name', 'ram_capacity', 'processor_brand', 'fast_charging',
    'primary_camera_rear', 'primary_camera_front', and 'extended_upto'.

    :param X: The input DataFrame.
    :type X: pd.DataFrame
    :return: The DataFrame with encoded features.
    :rtype: pd.DataFrame
    """
    logging.info("Encoding features.")
    logging.info(f"Original DataFrame shape: {X.shape}")
    X = X.copy()

    try:
        # resolution
        logging.debug("Encoding resolution.")
        X["resolution"] = np.select(
            [
                X["resolution"].str.extract(r"(\d+)").astype(float).iloc[:, 0] < 1080,
                (X["resolution"].str.extract(r"(\d+)").astype(float).iloc[:, 0] >= 1080)
                & (
                    X["resolution"].str.extract(r"(\d+)").astype(float).iloc[:, 0]
                    < 1440
                ),
                (X["resolution"].str.extract(r"(\d+)").astype(float).iloc[:, 0] >= 1440)
                & (
                    X["resolution"].str.extract(r"(\d+)").astype(float).iloc[:, 0]
                    < 2160
                ),
                X["resolution"].str.extract(r"(\d+)").astype(float).iloc[:, 0] >= 2160,
            ],
            ["HD Ready", "Full HD", "Quad HD", "Ultra HD"],
            default="Unknown",
        )
        logging.debug("Resolution encoded.")

        # brand_name
        logging.debug("Encoding brand_name.")
        X["brand_name"] = X["brand_name"].mask(
            X["brand_name"].map(X["brand_name"].value_counts()) < 10, "other"
        )
        logging.debug("brand_name encoded.")

        # ram_capacity
        logging.debug("Encoding ram_capacity.")
        X["ram_capacity"] = np.select(
            [
                X["ram_capacity"].astype(float) <= 4,
                X["ram_capacity"].astype(float) <= 8,
            ],
            ["Low", "Standard"],
            default="High",
        )
        logging.debug("ram_capacity encoded.")

        # processor_brand
        logging.debug("Encoding processor_brand.")
        X["processor_brand"] = X["processor_brand"].mask(
            X["processor_brand"].map(X["processor_brand"].value_counts()) < 5, "other"
        )
        logging.debug("processor_brand encoded.")

        # fast_charging
        logging.debug("Encoding fast_charging.")
        X["fast_charging"] = np.select(
            [
                X["fast_charging"] == 0.0,
                (X["fast_charging"] > 0.0) & (X["fast_charging"] <= 18),
                (X["fast_charging"] > 18) & (X["fast_charging"] <= 33),
                (X["fast_charging"] > 33) & (X["fast_charging"] <= 65),
                (X["fast_charging"] > 65) & (X["fast_charging"] <= 120),
            ],
            ["no", "basic", "moderate", "fast", "ultra fast"],
            default="extreme fast",
        )
        logging.debug("fast_charging encoded.")

        # primary_camera_rear
        logging.debug("Encoding primary_camera_rear.")
        X["primary_camera_rear"] = np.select(
            [
                X["primary_camera_rear"].astype(float) <= 8,
                (X["primary_camera_rear"].astype(float) > 8)
                & (X["primary_camera_rear"].astype(float) <= 16),
                (X["primary_camera_rear"].astype(float) > 16)
                & (X["primary_camera_rear"].astype(float) <= 48),
                (X["primary_camera_rear"].astype(float) > 48)
                & (X["primary_camera_rear"].astype(float) <= 64),
                (X["primary_camera_rear"].astype(float) > 64)
                & (X["primary_camera_rear"].astype(float) <= 108),
            ],
            ["basic", "standard", "good", "high-Resolution", "ultra-high"],
            default="extreme",
        )
        logging.debug("primary_camera_rear encoded.")

        # primary_camera_front
        logging.debug("Encoding primary_camera_front.")
        X["primary_camera_front"] = np.select(
            [
                X["primary_camera_front"].astype(float) <= 8,
                (X["primary_camera_front"].astype(float) > 8)
                & (X["primary_camera_front"].astype(float) <= 16),
                (X["primary_camera_front"].astype(float) > 16)
                & (X["primary_camera_front"].astype(float) <= 32),
                (X["primary_camera_front"].astype(float) > 32)
                & (X["primary_camera_front"].astype(float) <= 48),
            ],
            ["basic", "standard", "good", "high-resolution"],
            default="ultra high",
        )
        logging.debug("primary_camera_front encoded.")

        # extended_upto
        logging.debug("Encoding extended_upto.")
        X["extended_upto"] = np.select(
            [
                X["extended_upto"] == 0.0,
                X["extended_upto"].isin([64.0, 128.0, 256.0]),
                X["extended_upto"] == 512.0,
                X["extended_upto"] == 1024.0,
                X["extended_upto"] == 2048.0,
            ],
            ["no expansion", "small", "medium", "large", "very large"],
            default="unknown",
        )
        logging.debug("extended_upto encoded.")

        logging.info("Features encoded successfully.")
    except Exception as e:
        logging.error(f"Error encoding features: {e}")
        raise  # re-raise exception so it is not hidden.
    logging.info(f"Transformed DataFrame shape: {X.shape}")
    return X


def get_transformer() -> object:
    try:
        # missing value imputation pipeline
        missing_handler = Pipeline(
            [
                ("camera_dropna", DropNA(subset=["primary_camera_front"])),
                ("processor_brand", ProcessorBrandImputer()),
                ("num_cores_dropna", DropNA(subset=["num_cores"])),
                ("processor_speed_dropna", DropNA(subset=["processor_speed"])),
                ("extended_upto", ExtendedUptoImputer()),
                ("fast_charging", FastChargingImputer()),
                ("consistent_brands", ConsistentBrandsImputer()),
                ("higher_missing_brands", HigherMissingBrandsImputer()),
                ("lower_missing_brands", LowerMissingBrandsImputer()),
                ("single_record_brands", SingleRecordBrandsRemover()),
            ]
        )

        # feature construction pipeline
        feature_constructor = Pipeline(
            [
                ("performance_score", FunctionTransformer(compute_performance_score)),
                ("camera_quality", FunctionTransformer(compute_camera_quality)),
                (
                    "connectivity_features",
                    FunctionTransformer(compute_connectivity_features),
                ),
                (
                    "storage_expandability",
                    FunctionTransformer(compute_storage_expandability),
                ),
            ]
        )

        feature_encoder = ColumnTransformer(
            [
                ("standard_scaling", StandardScaler(), standard_scale_cols),
                ("minmax_scaling", MinMaxScaler(), minmax_scale_cols),
                (
                    "nominal_encoding",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    nominal_cols,
                ),
                (
                    "ordinal_encoding",
                    OrdinalEncoder(
                        categories=ordinal_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ordinal_cols,
                ),
            ]
        )

        preprocessor = Pipeline(
            [
                ("handle_missing", missing_handler),
                ("feature_construction", feature_constructor),
                ("high_cardinality", FunctionTransformer(encode_features)),
                ("feature_encoding", feature_encoder),
            ]
        )

        return preprocessor

    except Exception as e:
        raise SmartPhonesException(e, sys)
