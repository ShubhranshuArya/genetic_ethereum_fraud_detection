import streamlit as st
import requests
import json
import random  # Import the random module

# Set the URL of the MLflow prediction server
PREDICTION_URL = (
    "http://127.0.0.1:8000/invocations"  # Update with your actual prediction server URL
)

# Streamlit app title
st.title("Ethereum Fraud Detection")

# Input fields for the model features
avg_min_between_sent_tnx = st.number_input("Average Min Between Sent Transactions")
avg_min_between_received_tnx = st.number_input(
    "Average Min Between Received Transactions"
)
time_diff_between_first_and_last = st.number_input(
    "Time Diff Between First and Last (mins)"
)
sent_tnx = st.number_input("Sent Transactions")
received_tnx = st.number_input("Received Transactions")
number_of_created_contracts = st.number_input("Number of Created Contracts")
unique_received_from_addresses = st.number_input("Unique Received From Addresses")
unique_sent_to_addresses = st.number_input("Unique Sent To Addresses")
min_value_received = st.number_input("Min Value Received")
max_value_received = st.number_input("Max Value Received")
avg_val_received = st.number_input("Avg Value Received")
min_val_sent = st.number_input("Min Value Sent")
max_val_sent = st.number_input("Max Value Sent")
avg_val_sent = st.number_input("Avg Value Sent")
total_transactions = st.number_input(
    "Total Transactions (including tnx to create contract)"
)
total_ether_sent = st.number_input("Total Ether Sent")
total_ether_received = st.number_input("Total Ether Received")
total_ether_balance = st.number_input("Total Ether Balance")
total_erc20_tnxs = st.number_input("Total ERC20 Transactions")
erc20_total_ether_received = st.number_input("ERC20 Total Ether Received")
erc20_total_ether_sent = st.number_input("ERC20 Total Ether Sent")
erc20_total_ether_sent_contract = st.number_input("ERC20 Total Ether Sent Contract")
erc20_uniq_sent_addr = st.number_input("ERC20 Unique Sent Addresses")
erc20_uniq_rec_addr = st.number_input("ERC20 Unique Received Addresses")
erc20_uniq_rec_contract_addr = st.number_input(
    "ERC20 Unique Received Contract Addresses"
)
erc20_min_val_rec = st.number_input("ERC20 Min Value Received")
erc20_avg_val_rec = st.number_input("ERC20 Avg Value Received")
erc20_uniq_sent_token_name = st.number_input("ERC20 Unique Sent Token Name")

# Button to generate random values
if st.button("Generate Random Values"):
    avg_min_between_sent_tnx = random.uniform(0, 10)
    avg_min_between_received_tnx = random.uniform(0, 10)
    time_diff_between_first_and_last = random.uniform(0, 60)
    sent_tnx = random.randint(0, 100)
    received_tnx = random.randint(0, 100)
    number_of_created_contracts = random.randint(0, 50)
    unique_received_from_addresses = random.randint(0, 20)
    unique_sent_to_addresses = random.randint(0, 20)
    min_value_received = random.uniform(0, 5)
    max_value_received = random.uniform(5, 100)
    avg_val_received = random.uniform(0, 50)
    min_val_sent = random.uniform(0, 5)
    max_val_sent = random.uniform(5, 100)
    avg_val_sent = random.uniform(0, 50)
    total_transactions = random.randint(0, 200)
    total_ether_sent = random.uniform(0, 10)
    total_ether_received = random.uniform(0, 10)
    total_ether_balance = random.uniform(0, 10)
    total_erc20_tnxs = random.randint(0, 100)
    erc20_total_ether_received = random.uniform(0, 10)
    erc20_total_ether_sent = random.uniform(0, 10)
    erc20_total_ether_sent_contract = random.uniform(0, 10)
    erc20_uniq_sent_addr = random.randint(0, 20)
    erc20_uniq_rec_addr = random.randint(0, 20)
    erc20_uniq_rec_contract_addr = random.randint(0, 20)
    erc20_min_val_rec = random.uniform(0, 5)
    erc20_avg_val_rec = random.uniform(0, 5)
    erc20_uniq_sent_token_name = random.randint(1, 10)  # Assuming this is an integer

# Button to make prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "dataframe_records": [
            {
                "avg_min_between_sent_tnx": avg_min_between_sent_tnx,
                "avg_min_between_received_tnx": avg_min_between_received_tnx,
                "time_diff_between_first_and_last_(mins)": time_diff_between_first_and_last,
                "sent_tnx": sent_tnx,
                "received_tnx": received_tnx,
                "number_of_created_contracts": number_of_created_contracts,
                "unique_received_from_addresses": unique_received_from_addresses,
                "unique_sent_to_addresses": unique_sent_to_addresses,
                "min_value_received": min_value_received,
                "max_value_received": max_value_received,
                "avg_val_received": avg_val_received,
                "min_val_sent": min_val_sent,
                "max_val_sent": max_val_sent,
                "avg_val_sent": avg_val_sent,
                "total_transactions_(including_tnx_to_create_contract": total_transactions,
                "total_ether_sent": total_ether_sent,
                "total_ether_received": total_ether_received,
                "total_ether_balance": total_ether_balance,
                "total_erc20_tnxs": total_erc20_tnxs,
                "erc20_total_ether_received": erc20_total_ether_received,
                "erc20_total_ether_sent": erc20_total_ether_sent,
                "erc20_total_ether_sent_contract": erc20_total_ether_sent_contract,
                "erc20_uniq_sent_addr": erc20_uniq_sent_addr,
                "erc20_uniq_rec_addr": erc20_uniq_rec_addr,
                "erc20_uniq_rec_contract_addr": erc20_uniq_rec_contract_addr,
                "erc20_min_val_rec": erc20_min_val_rec,
                "erc20_avg_val_rec": erc20_avg_val_rec,
                "erc20_uniq_sent_token_name": erc20_uniq_sent_token_name,
            }
        ]
    }

    # Convert the input data to JSON format
    json_data = json.dumps(input_data)

    # Set the headers for the request
    headers = {"Content-Type": "application/json"}

    # Send the POST request to the server
    response = requests.post(PREDICTION_URL, headers=headers, data=json_data)

    # Check the response status code
    if response.status_code == 200:
        # If successful, print the prediction result
        prediction = response.json()
        st.success(f"Prediction: {prediction}")
    else:
        # If there was an error, print the status code and the response
        st.error(f"Error: {response.status_code}")
        st.error(response.text)
