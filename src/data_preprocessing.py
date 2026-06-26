import os
import sys
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Custom Exception
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = str(error_message)

    def __str__(self):
        return self.error_message


class DataTransformation:

    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def get_data_transformer_object(self):
        """
        Creates preprocessing pipeline
        """
        try:
            categorical_columns = [
                "fuel",
                "seller_type",
                "transmission",
                "owner",
                "Brand"
            ]

            numerical_columns = [
                "km_driven",
                "Car_Age"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "onehotencoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            drop="first",
                            sparse_output=False  # Add this to get dense arrays directly
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",
                     num_pipeline,
                     numerical_columns),

                    ("cat_pipeline",
                     cat_pipeline,
                     categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # ----------------------------
            # Feature Engineering
            # ----------------------------

            CURRENT_YEAR = 2024

            train_df["Car_Age"] = (
                CURRENT_YEAR - train_df["year"]
            )

            test_df["Car_Age"] = (
                CURRENT_YEAR - test_df["year"]
            )

            train_df["Brand"] = (
                train_df["name"]
                .str.split()
                .str[0]
                .str.lower()
            )

            test_df["Brand"] = (
                test_df["name"]
                .str.split()
                .str[0]
                .str.lower()
            )

            train_df.drop(
                columns=["year", "name"],
                inplace=True
            )

            test_df.drop(
                columns=["year", "name"],
                inplace=True
            )

            # ----------------------------
            # Target Column
            # ----------------------------

            target_column_name = "selling_price"

            X_train = train_df.drop(
                columns=[target_column_name]
            )

            y_train = train_df[target_column_name].values.ravel()

            X_test = test_df.drop(
                columns=[target_column_name]
            )

            y_test = test_df[target_column_name].values.ravel()

            preprocessor = self.get_data_transformer_object()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # Convert to dense arrays if they are sparse
            if hasattr(X_train_arr, "toarray"):
                X_train_arr = X_train_arr.toarray()
            if hasattr(X_test_arr, "toarray"):
                X_test_arr = X_test_arr.toarray()

            print("X_train_arr shape:", X_train_arr.shape)
            print("X_test_arr shape:", X_test_arr.shape)
            print("y_train shape:", y_train.shape)
            print("y_test shape:", y_test.shape)

            # Concatenate features with target
            train_arr = np.column_stack([X_train_arr, y_train])
            test_arr = np.column_stack([X_test_arr, y_test])

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            with open(
                self.preprocessor_obj_file_path,
                "wb"
            ) as file_obj:
                pickle.dump(
                    preprocessor,
                    file_obj
                )

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)