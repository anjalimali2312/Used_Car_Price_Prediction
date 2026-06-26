import sys
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.utils import load_object


# -----------------------------
# Custom Exception
# -----------------------------
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = str(error_message)

    def __str__(self):
        return self.error_message


# -----------------------------
# Model Evaluation
# -----------------------------
class ModelEvaluation:

    def evaluate_model(
        self,
        model_path,
        test_array
    ):

        try:

            # Load trained model
            model = load_object(
                model_path
            )

            # Split test data
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Predictions
            y_pred = model.predict(
                X_test
            )

            # Metrics
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

            print("\n" + "=" * 50)
            print("MODEL EVALUATION RESULTS")
            print("=" * 50)

            print(
                f"MAE      : {mae:.4f}"
            )

            print(
                f"RMSE     : {rmse:.4f}"
            )

            print(
                f"R² Score : {r2:.4f}"
            )

            print("=" * 50)

            return {
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2
            }

        except Exception as e:
            raise CustomException(
                e,
                sys
            )