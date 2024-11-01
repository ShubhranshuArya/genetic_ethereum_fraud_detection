from zenml import step
from src.missing_value_handling import (
    MissingValueHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
)
import pandas as pd


@step
def missing_value_handling_step(
    df: pd.DataFrame,
    strategy: str = "drop",
) -> pd.DataFrame:
    """
    Handles missing values using MissingValueHandler and the specified strategy
    with the Enum class 'MissingValuesHandlingStrategy'.

    Args:
        df (pd.DataFrame): The input DataFrame containing missing values.
        strategy (MissingValuesHandlingStrategy): The strategy to use for handling missing values.
            Default is MissingValuesHandlingStrategy.DROP.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled according to the specified strategy.

    Raises:
        ValueError: If an unsupported missing value handling strategy is provided.
    """

    if strategy == "drop":
        handler = MissingValueHandler(DropMissingValuesStrategy())
    elif strategy in [
        "mean",
        "median",
        "mode",
        "constant",
    ]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError(f"Unsupported missing value handling strategy: {strategy}")

    cleaned_df = handler.apply_strategy(df)
    return cleaned_df
