import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataTransformation

from src.data_preprocessing import DataTransformation
from src.model_trainer import ModelTrainer


# Custom exception
class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message

    def __str__(self):
        return self.error_message


class DataIngestion:
    """
    Handles:
    1. Reading dataset
    2. Train-Test Split
    3. Saving raw/train/test data
    """

    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            print("Reading dataset...")

            df = pd.read_csv("data/car_data.csv")

            os.makedirs("artifacts", exist_ok=True)

            # Save raw dataset
            df.to_csv(self.raw_data_path, index=False)

            print("Train-Test Split Started")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            train_set.to_csv(
                self.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.test_data_path,
                index=False,
                header=True
            )

            print("Data Ingestion Completed")

            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    print(f"Train Data Saved At : {train_data}")
    print(f"Test Data Saved At  : {test_data}")




if __name__ == "__main__":

    obj = DataIngestion()

    train_data_path, test_data_path = (
        obj.initiate_data_ingestion()
    )

    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
    )



if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformer = DataTransformation()

    train_arr, test_arr, _ = (
        transformer.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    trainer = ModelTrainer()

    trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )