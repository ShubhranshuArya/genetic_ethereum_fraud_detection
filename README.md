![project_poster](https://github.com/user-attachments/assets/cc38a820-0d00-4a85-ae6e-a16769f12ea2)

This project provides a solution to detect fraudulent transactions within the Ethereum network by leveraging the state-of-the-art genetic algorithm to optimize a neural network for modelling. It combines advanced data processing with fraud detection strategies, offering a robust pipeline for analyzing Ethereum transactions and identifying potential fraud in real-time.

## 📙 Table of Contents
- [Technical Summary](#-technical-summary)
- [Developed Pipelines](#%EF%B8%8F-developed-pipelines)
- [Project Structure](#-project-structure)
- [Setup and Installation](#-setup-and-installation)
- [ZenML Integration](#-zenml-installation)
- [Data Preparation](#-data-preparation)
- [Running the Project](#-running-the-project)
- [Contributing](#-contributing)


## 👇 Technical Summary

This project uses a genetic algorithm to enhance fraud detection in Ethereum transactions. The core of the project consists of data ingestion, preprocessing, model training, and a deployment pipeline:

1. **Data Ingestion**: Parses and ingests Ethereum transaction data from PostgreSQL.
2. **Feature Engineering**: Extracts and processes relevant features from the raw data, making it suitable for model training.
3. **Model Training**: A genetic algorithm with optimised hyperparameters for the neural network to improve fraud detection accuracy.
4. **Deployment and Prediction**: The trained model is deployed using Streamlit to provide a user-friendly interface for predictions.

By combining genetic algorithms and machine learning, this project aims to achieve high accuracy in detecting potentially fraudulent transactions, aiding in the security of blockchain transactions.


## ⚡️ Developed Pipelines
### -> Continuous Deployment Pipeline
This pipeline is designed to automate the process of deploying trained machine-learning models. It ensures that the latest and most effective models are continuously deployed, facilitating seamless updates and improvements in the model serving process.
![deployment_pipeline](https://github.com/user-attachments/assets/46d25235-00ed-4876-91e0-35edaca7d996)


### -> Inference Pipeline
This pipeline is focused on executing batch inference jobs using the continuous deployed model. This pipeline is essential for processing large volumes of data efficiently, allowing users to obtain predictions in a batch manner.
![Untitled design](https://github.com/user-attachments/assets/174f1086-28e4-402b-bc30-7fcbfb14a58f)


## 🔥 Project Structure

```plaintext
├── data/                   # Directory for Ethereum transaction data files (CSV or ZIP format)
├── models/                 # Contains trained models and saved artifacts
├── notebooks/              # Jupyter notebooks for experimentation and visualization
├── scripts/                # Scripts for data processing and feature engineering
├── app.py                  # Streamlit application for interactive prediction
├── requirements.txt        # List of dependencies
├── run_pipeline.py         # Main script for data ingestion and model training
├── run_deployment.py       # Script for deployment setup
└── README.md               # Project documentation
```


## 🧑‍💻 Setup and Installation
1. Clone the Repository
2. Set Up a Virtual Environment
```
python -m venv venv
```
3. Activate the virtual environment
```
## On macOS and Linux:
source venv/bin/activate
```
```
## On Windows:
venv\Scripts\activate
```
4. Install Dependencies
```
pip install -r requirements.txt
```


## 🧘 ZenML Installation
1. Install ZenML - https://docs.zenml.io/getting-started/installation 

2. Install some integrations using ZenML:
```
zenml integration install mlflow -y
```

3. Register mlflow in the stack:
```
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```


## 📚 Data Preparation

- To be added


## 🏃 Running the Project
The project can be run in distinct stages: data ingestion, model training, deployment, and prediction.

Step 1: Model Training
This script iterates through genetic algorithm cycles, optimizing model hyperparameters for the best fraud detection performance. Once the data is ingested, initiate model training with the genetic algorithm optimization by executing:
```
python run_pipeline.py
```

Step 2: Deployment
To deploy the model and set up the Streamlit interface, run:
```
python run_deployment.py
```

Step 3: Prediction Interface
Start the Streamlit app to make predictions:
```
streamlit run app.py
```

Once the application is running, navigate to the provided local URL (typically http://localhost:8501) to access the web interface. Here, you can input transaction data and obtain predictions on the likelihood of fraud.


## 🤝 Contributing
Whether you're interested in enhancing the genetic algorithm, refining the fraud detection model, or improving the user interface, your input is invaluable.

To contribute:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit them.
- Push to the branch (git push origin feature-branch).
- Open a pull request.

Together, we can strengthen the detection of fraudulent transactions in the Ethereum space. Thank you for your interest and support!
