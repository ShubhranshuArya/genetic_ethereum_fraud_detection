import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from keras.models import load_model
from lightgbm import LGBMRegressor


# Abstract class for Model Building Strategy
class DataModellingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> RegressorMixin:
        """
        Builds and trains the GA-CS model using the provided training data.

        This method is responsible for creating the GA-CS model, fitting it to the training data, and returning the trained model. The model should be an instance of `RegressorMixin` from scikit-learn.

        Parameters:
            X_train (pd.DataFrame): The feature data for training the model.
            y_train (pd.Series): The target data for training the model.

        Returns:
            RegressorMixin: The trained regression model.
        """
        pass


# Concrete class to implement the GACS Algorithm.
class GACSModellingStrategy(DataModellingStrategy):
    def build_and_train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RegressorMixin:
        """
        Builds and trains the GACS model using the provided training data.

        Parameters:
            X_train (pd.DataFrame): The feature data for training the model.
            y_train (pd.Series): The target data for training the model.

        Returns:
            Pipeline: The trained pipeline containing the GACS model.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Loading the genetic model.")

        try:
            # Load the pre-trained model
            gacs_model = load_model(
                "/Users/botcoder/MyDocs/AiLearn/Antern Learn/project/ethereum-fraud-detection/pre_trained_models/genetic_model.h5"
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Create a pipeline with the loaded model only
        pipeline = Pipeline([("model", gacs_model)])

        logger.info("Fitting the pipeline to the training data.")
        pipeline.fit(X_train, y_train)
        logger.info("Model training completed successfully.")

        return pipeline


# Concrete class to implement the LGBM Algorithm.
class LGBMModellingStrategy(DataModellingStrategy):
    def build_and_train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RegressorMixin:
        """
        Builds and trains the LGBM model using the provided training data.

        Parameters:
            X_train (pd.DataFrame): The feature data for training the model.
            y_train (pd.Series): The target data for training the model.

        Returns:
            Pipeline: The trained pipeline containing the LGBM model.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Building the LGBM model.")

        try:
            # Initialize the LGBM Regressor
            lgbm_model = LGBMRegressor()

            # Create a pipeline with the LGBM model
            pipeline = Pipeline([("model", lgbm_model)])

            logger.info("Fitting the pipeline to the training data.")
            pipeline.fit(X_train, y_train)
            logger.info("Model training completed successfully.")

            return pipeline
        except Exception as e:
            logger.error(f"Error building or training the LGBM model: {e}")
            raise


# Context class for Model Building
class DataModeller:
    def __init__(self, strategy: DataModellingStrategy):
        """
        Initializes the DataModellingContext with the provided strategy.

        Parameters:
            strategy (DataModellingStrategy): The strategy to be used for data modelling.
        """
        self._strategy = strategy

    @property
    def strategy(self):
        """
        Getter method for the strategy attribute.

        Returns:
            DataModellingStrategy: The strategy to be used for data modelling.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DataModellingStrategy):
        """
        Setter method for the strategy attribute.

        Parameters:
            strategy (DataModellingStrategy): The strategy to be used for data modelling.
        """
        self._strategy = strategy

    def build_and_train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> RegressorMixin:
        """
        Builds and trains the model using the selected strategy.

        Parameters:
            X_train (pd.DataFrame): The feature data for training the model.
            y_train (pd.Series): The target data for training the model.

        Returns:
            RegressorMixin: The trained regression model.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)
