# Predictive Analysis for Manufacturing Operations

## Project Overview

This project provides an API for predictive analysis of manufacturing operations. It allows users to upload a dataset, train a machine learning model, and make predictions about machine downtime based on manufacturing data.

The dataset used in this project is Machine_Downtime.csv, which contains the following columns:
-Machine_ID 

-Assembly_Line_No (Categorical): ['Shopfloor-L1', 'Shopfloor-L2', 'Shopfloor-L3']

-Hydraulic_Pressure(bar)

-Coolant_Pressure(bar)

-Air_System_Pressure(bar)

-Coolant_Temperature

-Hydraulic_Oil_Temperature(?C)

-Spindle_Bearing_Temperature(?C)

-Spindle_Vibration(?m)

-Tool_Vibration(?m)

-Spindle_Speed(RPM)

-Voltage(volts)

-Torque(Nm)

-Cutting(kN)

-Downtime (Categorical): ['Machine_Failure', 'No_Machine_Failure']


The API provides three endpoints:
/upload: Upload a CSV file.
/train: Train a machine learning model.
/predict: Make predictions using the trained model.

## Instructions to Set Up and Run the API

### Prerequisites
Python 3.8 or higher
pip (Python package manager)
Flask
scikit-learn
pandas

### Setup and Execution
Install the required libraries: pip install flask scikit-learn pandas
Start the Flask server: python Predictive_Analysis_Endpoint.py
The API will run on http://127.0.0.1:5000 by default.

## API Endpoints

1. Upload Endpoint

URL: /upload
Method: POST

Description: Upload a CSV file containing manufacturing data.

Request Example:
curl -X POST -F "file=@Machine_Downtime.csv" http://127.0.0.1:5000/upload

Response Example:
{
    "message": "File uploaded successfully"
}

2. Train Endpoint

URL: /train
Method: POST

Description: Train a machine learning model on the uploaded dataset.

Request Example:
curl -X POST http://127.0.0.1:5000/train

Response Example:
{
    "accuracy": 0.962,
    "f1_score": 0.96201,
    "precision": 0.96202,
    "recall": 0.962
}

3. Predict Endpoint

URL: /predict
Method: POST

Description: Make predictions based on input data.

Request Example 1:
curl -X POST -H "Content-Type: application/json" -d '{
    "Assembly_Line_No": "Shopfloor-L1",
    "Hydraulic_Pressure(bar)": 120.5,
    "Coolant_Pressure(bar)": 30.2,
    "Air_System_Pressure(bar)": 45.6,
    "Coolant_Temperature": 85.3,
    "Hydraulic_Oil_Temperature(?C)": 65.8,
    "Spindle_Bearing_Temperature(?C)": 72.1,
    "Spindle_Vibration(?m)": 0.02,
    "Tool_Vibration(?m)": 0.03,
    "Spindle_Speed(RPM)": 1500,
    "Voltage(volts)": 220,
    "Torque(Nm)": 55.5,
    "Cutting(kN)": 25.8
}' http://127.0.0.1:5000/predict

Reponse Example 1: 
{
    "Downtime": "Machine_Failure"
}

Request Example 2:
curl -X POST -H "Content-Type: application/json" -d '{
    "Assembly_Line_No": "Shopfloor-L1",
    "Hydraulic_Pressure(bar)": 135.811931,
    "Coolant_Pressure(bar)": 5.09404,
    "Air_System_Pressure(bar)": 6.818871,
    "Coolant_Temperature": 21.2,
    "Hydraulic_Oil_Temperature(?C)": 44.4,
    "Spindle_Bearing_Temperature(?C)": 27.4,
    "Spindle_Vibration(?m)": 0.762,
    "Tool_Vibration(?m)": 30.761,
    "Spindle_Speed(RPM)": 15217.0,
    "Voltage(volts)": 236.0,
    "Torque(Nm)": 23.318615,
    "Cutting(kN)": 1.88
}' http://127.0.0.1:5000/predict

Reponse Example 2: 
{
    "Downtime": "No_Machine_Failure"
}

## Notes

Make sure the dataset includes all required columns.
Ensure that the dataset format matches the expected structure to avoid errors.
For unseen labels in Assembly_Line_No or Downtime, update the encoders accordingly.

