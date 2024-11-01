import json
import numpy as np
import pandas as pd
from zenml import step

from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop("columns", None)  # Remove 'columns' if it's present
    data.pop("index", None)  # Remove 'index' if it's present

    expected_columns = [
        "avg_min_between_sent_tnx",
        "avg_min_between_received_tnx",
        "time_diff_between_first_and_last_(mins)",
        "sent_tnx",
        "received_tnx",
        "number_of_created_contracts",
        "unique_received_from_addresses",
        "unique_sent_to_addresses",
        "min_value_received",
        "max_value_received",
        "avg_val_received",
        "min_val_sent",
        "max_val_sent",
        "avg_val_sent",
        "total_transactions_(including_tnx_to_create_contract",
        "total_ether_sent",
        "total_ether_received",
        "total_ether_balance",
        "total_erc20_tnxs",
        "erc20_total_ether_received",
        "erc20_total_ether_sent",
        "erc20_total_ether_sent_contract",
        "erc20_uniq_sent_addr",
        "erc20_uniq_rec_addr",
        "erc20_uniq_rec_contract_addr",
        "erc20_min_val_rec",
        "erc20_avg_val_rec",
        "erc20_uniq_sent_token_name",
    ]

    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction
