import streamlit as st
import requests

# Define the API URL
API_URL = "http://127.0.0.1:8000/predict"

# Title of the Streamlit App
st.title("Iris Model Inference Dashboard")

# Input fields for the four features
feature1 = st.number_input("Feature 1 (Sepal Length)", value=5.1)
feature2 = st.number_input("Feature 2 (Sepal Width)", value=3.5)
feature3 = st.number_input("Feature 3 (Petal Length)", value=1.4)
feature4 = st.number_input("Feature 4 (Petal Width)", value=0.2)

# Button to get predictions
if st.button("Predict"):
    # Prepare the input payload
    input_data = {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4,
    }

    # Call the API
    response = requests.post(API_URL, json=input_data)
    if response.status_code == 200:
        # Display the predicted class name
        result = response.json()
        class_name = result['class_name']
        st.success(f"Predicted Class: {class_name}")
    else:
        # Display the error message
        st.error(f"Error: {response.text}")
