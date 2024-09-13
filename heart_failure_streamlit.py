import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

# Load the model
model = load('heart_failure_model.joblib')

# Create a user input field for the feature
user_input = st.number_input('Enter age:', min_value=0, max_value=120, step=1)

# Create a DataFrame with the same column name as used in training
input_df = pd.DataFrame([[user_input]], columns=['Age'])

# Predict the probability of heart disease
if st.button('Predict Heart Disease Probability'):
    predicted_probability = model.predict(input_df)
    predicted_percentage = predicted_probability[0] * 100

    # Determine the prediction result
    threshold = 0.5  # Define your threshold (50% here)
    prediction_result = "Heart Disease" if predicted_probability[0] >= threshold else "No Heart Disease"

    # Display the results
    st.write(f"Predicted Heart Disease Probability: {predicted_percentage:.2f}%")
    st.write(f"Prediction: {prediction_result}")
