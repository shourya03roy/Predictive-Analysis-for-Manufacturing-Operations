'''
This project provides an API for predictive analysis of manufacturing operations. 
It allows users to upload a dataset, train a machine learning model, and make predictions about machine downtime based on manufacturing data.
'''

# importing libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Initializing Flask application
app = Flask(__name__)

# Global variables for dataset and model
df = None
model = None
scaler = None
aln_encoder = LabelEncoder()
downtime_encoder = LabelEncoder()

# Endpoint to upload the dataset
@app.route('/upload', methods=['POST'])
def upload_file():
    global df

    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        return jsonify({'message': 'File uploaded successfully'}), 200
    
    except Exception as e:
        # Handle any errors during file upload
        return jsonify({'error': str(e)}), 500

# Endpoint to train the machine learning model
@app.route('/train', methods=['POST'])
def train_model():
    global model, df, scaler

    # Ensure a dataset has been uploaded
    if df is None:  
        return jsonify({'error': 'No data uploaded'}), 400
    
    try:

        # Remove Machine_ID
        df = df.drop(columns = ['Machine_ID'])

        # Encode categorical columns ('Assembly_Line_No' and 'Downtime')
        df['Assembly_Line_No'] = aln_encoder.fit_transform(df['Assembly_Line_No'])
        df['Downtime'] = downtime_encoder.fit_transform(df['Downtime'])

        # Separate features (X) and target (y)
        X = df.drop(columns=['Downtime'])
        y = df['Downtime']

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Return evaluation metrics as JSON response
        return jsonify({
            "accuracy": round(accuracy, 5),
            "precision": round(precision, 5),
            "recall": round(recall, 5),
            "f1_score": round(f1, 5)
        })
    
    except Exception as e:
        # Handle any errors during training
        return jsonify({'error': str(e)}), 500

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    
    # Ensure the model and scaler are initialized (trained)
    if model is None or scaler is None:
        return jsonify({'error': 'Model not trained yet'}), 400

    try:
        # Get input JSON for prediction
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data]) # Convert input to DataFrame

       # Encode categorical features in the input data
        input_df['Assembly_Line_No'] = aln_encoder.transform(input_df['Assembly_Line_No'])

        # Scale the input features using the trained scaler
        input_df_scaled = scaler.transform(input_df)

        # Make prediction using the trained model
        prediction = model.predict(input_df_scaled)

        # Decode the prediction back to the original 'Downtime' labels
        result = downtime_encoder.inverse_transform(prediction)
        return jsonify({'Downtime': result[0]}), 200
    
    except Exception as e:
        # Handle errors during prediction
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
