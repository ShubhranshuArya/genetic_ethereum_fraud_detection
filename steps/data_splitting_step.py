from typing import Tuple

import pandas as pd
from zenml import step

from src.data_splitting import DataSplitter, SimpleTrainTestSplitStrategy


@step
def data_splitter_step(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Splits the data into training and testing sets using DataSplitter and a chosen strategy.
    """
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)

    return X_train, X_test, y_train, y_test
