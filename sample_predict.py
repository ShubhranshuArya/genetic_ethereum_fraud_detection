import json

import requests

# URL of the MLflow prediction server
url = "http://127.0.0.1:8000/invocations"

# Sample input data for prediction
# Replace the values with the actual features your model expects
input_data = {
    "dataframe_records": [
        {
            "avg_min_between_sent_tnx": 9571.62,
            "avg_min_between_received_tnx": 92.65,
            "time_diff_between_first_and_last_(mins)": 69410.25,
            "sent_tnx": 7,
            "received_tnx": 26,
            "number_of_created_contracts": 0,
            "unique_received_from_addresses": 24,
            "unique_sent_to_addresses": 4,
            "min_value_received": 0.001312,
            "max_value_received": 4.9,
            "avg_val_received": 1.063073,
            "min_val_sent": 0.5,
            "max_val_sent": 12.003871,
            "avg_val_sent": 4.02341,
            "total_transactions_(including_tnx_to_create_contract": 33,
            "total_ether_sent": 28.16387078,
            "total_ether_received": 27.63989778,
            "total_ether_balance": -0.5239,
            "total_erc20_tnxs": 2.0,
            "erc20_total_ether_received": 0.0002,
            "erc20_total_ether_sent": 0.0,
            "erc20_total_ether_sent_contract": 0.0,
            "erc20_uniq_sent_addr": 0.0,
            "erc20_uniq_rec_addr": 2.0,
            "erc20_uniq_rec_contract_addr": 2.0,
            "erc20_min_val_rec": 0.0,
            "erc20_avg_val_rec": 0.0,
            "erc20_uniq_sent_token_name": 0.0,
        }
    ]
}

# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {"Content-Type": "application/json"}

# Send the POST request to the server
response = requests.post(url, headers=headers, data=json_data)

# Check the response status code
if response.status_code == 200:
    # If successful, print the prediction result
    prediction = response.json()
    print("Prediction:", prediction)
else:
    # If there was an error, print the status code and the response
    print(f"Error: {response.status_code}")
    print(response.text)
