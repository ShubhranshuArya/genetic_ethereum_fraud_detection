import streamlit as st
import requests
import json
import random

# Set your model endpoint here
PREDICTION_URL = (
    "https://your_model_server_url/predict"  # Replace with the actual endpoint
)

# Title for the Streamlit app
st.title("Ethereum Fraud Detection")

# Define feature names for reference
feature_names = [
    "avg_min_between_sent_tnx",
    "avg_min_between_received_tnx",
    "time_diff_between_first_and_last_(mins)",
    "sent_tnx",
    "received_tnx",
    "number_of_created_contracts",
    "max_value_received",
    "avg_val_received",
    "avg_val_sent",
    "total_ether_sent",
    "total_ether_balance",
    "erc20_total_ether_received",
    "erc20_total_ether_sent",
    "erc20_total_ether_sent_contract",
    "erc20_uniq_sent_addr",
    "erc20_uniq_rec_token_name",
]

# Fraud likely values dataset
fraud_likely_data = {
    "avg_min_between_sent_tnx": [163.26, 0.0, 6.33, 4.31, 25.25],
    "avg_min_between_received_tnx": [0.3, 0.0, 827.0, 18491.47, 0.45],
    "time_diff_between_first_and_last_(mins)": [
        327.12,
        4.17,
        127797.32,
        73983.1,
        76.65,
    ],
    "sent_tnx": [2, 1, 200, 4, 3],
    "received_tnx": [2, 1, 153, 4, 2],
    "number_of_created_contracts": [0, 0, 0, 0, 0],
    "max_value_received": [55.936436, 0.847878, 42.636256, 1.838791, 82.976949],
    "avg_val_received": [50.5, 0.847878, 16.394741, 0.515606, 50.5],
    "avg_val_sent": [50.499508, 0.847406, 12.541513, 0.514998, 33.739343],
    "total_ether_sent": [
        100.9990158,
        0.847406144,
        2508.30261,
        2.059993377,
        101.2180288,
    ],
    "total_ether_balance": [
        0.000984218,
        0.000471636,
        0.092767484,
        0.002429362,
        -0.218028773,
    ],
    "erc20_total_ether_received": [0.0, 0.0, 0.0, 0.0, 0.0],
    "erc20_total_ether_sent": [0.0, 0.0, 0.0, 0.0, 0.0],
    "erc20_total_ether_sent_contract": [0.0, 0.0, 0.0, 0.0, 0.0],
    "erc20_uniq_sent_addr": [0.0, 0.0, 0.0, 0.0, 0.0],
    "erc20_uniq_rec_token_name": [0.0, 0.0, 0.0, 0.0, 0.0],
}

# Initialize fraud data index for cycling
fraud_index = 0

# Input fields for each feature
st.markdown("### Input the values or generate random/fraud-likely values")

inputs = {}
for name in feature_names:
    inputs[name] = st.number_input(
        name.replace("_", " ").capitalize(), value=0.0, step=0.01
    )

# Generate random values button
if st.button("Generate Random Values"):
    for name in inputs:
        if "tx" in name:
            inputs[name] = random.randint(0, 100)
        else:
            inputs[name] = round(random.uniform(0, 100), 2)
    st.experimental_rerun()

# Use fraud-likely values button
if st.button("Fraud Likely Values"):
    values = fraud_likely_data[fraud_index]
    for i, name in enumerate(inputs):
        inputs[name] = values[i]
    fraud_index = (fraud_index + 1) % len(fraud_likely_data)
    st.experimental_rerun()

# Predict button
if st.button("Predict Fraud Probability"):
    # Prepare data as JSON
    input_data = {name: inputs[name] for name in inputs}
    json_data = json.dumps({"dataframe_records": [input_data]})

    # Make prediction request
    headers = {"Content-Type": "application/json"}
    response = requests.post(PREDICTION_URL, headers=headers, data=json_data)

    # Display prediction result
    if response.status_code == 200:
        prediction = response.json().get(
            "probability", 0
        )  # Adjust if API returns different structure
        fraud_percentage = float(prediction) * 100
        st.success(f"Prediction: {fraud_percentage:.2f}% likelihood of fraud")
    else:
        st.error("Prediction request failed")
        st.error(f"Error: {response.status_code}")
        st.error(response.text)
