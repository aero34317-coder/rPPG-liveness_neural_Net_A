import json
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

# Load model
def load_model(model_path):
    with open(model_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    return model

# Load scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'r') as json_file:
        scaler_info = json.load(json_file)
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_info['mean'])
    scaler.scale_ = np.array(scaler_info['scale'])
    return scaler

# Preprocess input data
def preprocess_input(data, scaler):
    data = np.array(data).reshape(1, -1)  # Reshape for scaler
    return scaler.transform(data)

# Make predictions
def make_prediction(model, data):
    return model.predict(data)

# Main inference function
def inference(model_path, scaler_path, input_data):
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    processed_data = preprocess_input(input_data, scaler)
    prediction = make_prediction(model, processed_data)
    return prediction

# Example of usage
if __name__ == "__main__":
    model_path = 'model.json'
    scaler_path = 'scaler.json'
    input_data = [/* your input data here */]
    result = inference(model_path, scaler_path, input_data)
    print(f"Prediction: {result}")
