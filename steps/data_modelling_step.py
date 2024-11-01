from zenml import step, Model, ArtifactConfig
from zenml.client import Client

# from typing import Annotated
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow

from src.data_modelling import (
    DataModeller,
    GACSModellingStrategy,
    LGBMModellingStrategy,
)

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="ethereum_fraud_detector",
    version=None,
    license="Apache 2.0",
    description="Predicts if an ETH transaction is fraudulent",
)


@step(
    enable_cache=False,
    experiment_tracker=experiment_tracker.name,
    model=model,
)
def data_modelling_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    This step is responsible for building and training a machine learning model to predict Ethereum transaction fraud.
    It utilizes the GACSModellingStrategy for model building and training, and leverages MLflow for experiment tracking.
    """
    data_modeller = DataModeller(LGBMModellingStrategy())

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active
        logging.info("Starting a new MLflow run for model training.")

    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()
        logging.info(
            "Enabling autologging for scikit-learn to capture model metrics and parameters."
        )

        model_pipeline = data_modeller.build_and_train_model(
            X_train=X_train,
            y_train=y_train,
        )
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()
        logging.info("Ending the MLflow run.")

    return model_pipeline
