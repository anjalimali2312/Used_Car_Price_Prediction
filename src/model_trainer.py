import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.utils import (save_object,evaluate_models)

from xgboost import XGBRegressor

from src.utils import evaluate_models

from src.utils import evaluate_models

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = str(error_message)

    def __str__(self):
        return self.error_message

class ModelTrainer:

    def __init__(self):

        self.trained_model_file_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

    def evaluate_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        models
    ):

        model_report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)

            model_report[model_name] = r2

        return model_report

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            print("Model Training Started")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Linear Regression":
                    LinearRegression(),

                "KNN":
                    KNeighborsRegressor(
                        n_neighbors=5
                    ),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=-1
                    ),

                "XGBoost":
                    XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1
                    )
            }

            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models
            )

            print("\nModel Performance:")
            for model_name, score in model_report.items():
                print(
                    f"{model_name} : "
                    f"{round(score,4)}"
                )

            best_model_score = max(
                sorted(model_report.values())
            )

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[
                best_model_name
            ]

            print(
                f"\nBest Model : "
                f"{best_model_name}"
            )

            print(
                f"R² Score : "
                f"{round(best_model_score,4)}"
            )

            if best_model_score < 0.6:
                raise Exception(
                    "No good model found."
                )

            os.makedirs(
                "artifacts",
                exist_ok=True
            )

            save_object(
               self.trained_model_file_path,best_model
            )

            y_pred = best_model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )

            r2 = r2_score(
                y_test,
                y_pred
            )

            print("\nFinal Metrics")
            print(
                f"MAE  : {round(mae,4)}"
            )
            print(
                f"RMSE : {round(rmse,4)}"
            )
            print(
                f"R²   : {round(r2,4)}"
            )

            return r2

        except Exception as e:
            raise CustomException(
                e,
                sys
            )