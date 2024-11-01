from abc import ABC, abstractmethod
import pandas as pd
import logging


# Abstract base class for handling missing values.
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete strategy to Drop missing values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
            axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        """
        self.axis = axis

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis}")
        cleaned_df = df.dropna(axis=self.axis)
        logging.info("Missing values dropped.")
        return cleaned_df


# Concrete strategy to fill missing values with statistical measures
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="median", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method.

        Parameters:
            method (str): The statistical measure to use for filling missing values.
                          Options: 'median', 'mean', 'mode' or 'constant'. Default is 'median'.
            fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame with the specified statistical measure.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values with {self.method}")

        df_cleaned = df.copy()
        if self.method == "median":
            numerical_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numerical_columns] = df_cleaned[numerical_columns].fillna(
                df[numerical_columns].median()
            )
        elif self.method == "mean":
            numerical_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numerical_columns] = df_cleaned[numerical_columns].fillna(
                df[numerical_columns].mean()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(
                f"Unknown method '{self.method}'. No missing values handled."
            )

        logging.info("Missing values filled.")
        return df_cleaned


# Context class for missing value handling
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Set the strategy for handling missing values.

        Parameters:
            strategy (MissingValueHandlingStrategy): The strategy to use for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def apply_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame using the current strategy.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values handled according to the strategy.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df)


# Usage example:
if __name__ == "__main__":
    # Create a sample DataFrame with missing values
    # df = pd.DataFrame(
    #     {"A": [1, 2, None, 4], "B": [5, None, 7, 8], "C": [9, 10, 11, None]}
    # )

    # # Create a MissingValueHandler
    # handler = MissingValueHandler()

    # # Set and use different strategies
    # handler.set_strategy(FillMissingValuesStrategy(method="mean"))
    # df_mean_filled = handler.handle(df)
    # print("Mean-filled DataFrame:")
    # print(df_mean_filled)

    # handler.set_strategy(FillMissingValuesStrategy(method="median"))
    # df_median_filled = handler.handle(df)
    # print("\nMedian-filled DataFrame:")
    # print(df_median_filled)

    # handler.set_strategy(FillMissingValuesStrategy(method="constant", fill_value=0))
    # df_constant_filled = handler.handle(df)
    # print("\nConstant-filled DataFrame:")
    # print(df_constant_filled)
    pass
