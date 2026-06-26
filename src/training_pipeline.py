from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataTransformation
from src.model_trainer import ModelTrainer
from src.model_evaluation import ModelEvaluation


def main():

    print("=" * 60)
    print("USED CAR PRICE PREDICTION PIPELINE STARTED")
    print("=" * 60)

    # ---------------------------------
    # Data Ingestion
    # ---------------------------------

    data_ingestion = DataIngestion()

    train_path, test_path = (
        data_ingestion.initiate_data_ingestion()
    )

    print("\n✅ Data Ingestion Completed")

    # ---------------------------------
    # Data Transformation
    # ---------------------------------

    data_transformation = DataTransformation()

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    print("✅ Data Transformation Completed")

    # ---------------------------------
    # Model Training
    # ---------------------------------

    model_trainer = ModelTrainer()

    model_score = (
        model_trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )
    )

    print("✅ Model Training Completed")

    # ---------------------------------
    # Model Evaluation
    # ---------------------------------

    model_evaluation = ModelEvaluation()

    evaluation_report = (
        model_evaluation.evaluate_model(
            model_path="artifacts/model.pkl",
            test_array=test_arr
        )
    )

    print("✅ Model Evaluation Completed")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFinal Results")

    print(
        f"MAE      : "
        f"{evaluation_report['MAE']:.4f}"
    )

    print(
        f"RMSE     : "
        f"{evaluation_report['RMSE']:.4f}"
    )

    print(
        f"R² Score : "
        f"{evaluation_report['R2 Score']:.4f}"
    )


if __name__ == "__main__":
    main()