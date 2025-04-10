from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initializing API
app = FastAPI()

# Load the trained model with error handling
try:
    with open("trained_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError("Model file not found. Please ensure 'trained_model.pkl' is in the correct directory.")

# Define the input schema for the API
class ModelInput(BaseModel):
    feature1:  float
    feature2:  float
    feature3:  float
    feature4:  float

@app.post("/predict")
def predict(input_data: ModelInput):
    # Convert input data to a NumPy array
    features = np.array([[input_data.feature1, input_data.feature2, input_data.feature3, input_data.feature4]])
    logging.info(f"Received input: {features}")

    # Make predictions
    prediction = model.predict(features)
    logging.info(f"Prediction result: {prediction}")

    # Map the predicted class to class names
    class_mapping = {
        0: "Class 0 (Iris-setosa)",
        1: "Class 1 (Iris-versicolor)",
        2: "Class 2 (Iris-virginica)"
    }
    class_name = class_mapping[int(prediction[0])]

    return {"class_name": class_name}  # Return class_name in the response

@app.get("/health")
def health_check():
    return {"status": "API is up and running!"}
