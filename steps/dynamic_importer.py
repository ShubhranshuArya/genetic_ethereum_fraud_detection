from zenml import step
import pandas as pd


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # In a real-world scenario, this could be an API call, database query, or loading from a file.

    data = {
        "avg_min_between_sent_tnx": [9571.62, 41.49],
        "avg_min_between_received_tnx": [92.65, 3394.1],
        "time_diff_between_first_and_last_(mins)": [69410.25, 1284742.27],
        "sent_tnx": [7, 370],
        "received_tnx": [26, 374],
        "number_of_created_contracts": [0, 0],
        "unique_received_from_addresses": [24, 4],
        "unique_sent_to_addresses": [4, 3],
        "min_value_received": [0.001312, 0.044836],
        "max_value_received": [4.9, 5.984678],
        "avg_val_received": [1.063073, 0.892415],
        "min_val_sent": [0.5, 0.042653],
        "max_val_sent": [12.003871, 5.9813],
        "avg_val_sent": [4.02341, 0.899],
        "total_transactions_(including_tnx_to_create_contract": [33, 744],
        "total_ether_sent": [28.16387078, 332.761830],
        "total_ether_received": [27.63989778, 333.7633261000001],
        "total_ether_balance": [-0.5239, 1.00149532],
        "total_erc20_tnxs": [2.0, 0.0],
        "erc20_total_ether_received": [0.0002, 0.0],
        "erc20_total_ether_sent": [0.0, 0.0],
        "erc20_total_ether_sent_contract": [0.0, 0.0],
        "erc20_uniq_sent_addr": [0.0, 0.0],
        "erc20_uniq_rec_addr": [2.0, 0.0],
        "erc20_uniq_rec_contract_addr": [2.0, 0.0],
        "erc20_min_val_rec": [0.0, 0.0],
        "erc20_avg_val_rec": [0.0, 0.0],
        "erc20_uniq_sent_token_name": [0.0, 0.0],
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    return json_data
