import streamlit as st
from joblib import load
import pandas as pd
import numpy as np

# Load the models and preprocessing steps
knn = load('knn_model.joblib')
ann = load('ann_model.joblib')
svm = load('svm_model.joblib')
scaler = load('scaler.joblib')
label_encoder = load('label_encoder.joblib')

# Create a user input field for the features
age = st.number_input('Enter age:', min_value=0, max_value=120, step=1)
resting_bp = st.number_input('Resting Blood Pressure:', min_value=0, step=1)
cholesterol = st.number_input('Cholesterol Level:', min_value=0, step=1)
max_hr = st.number_input('Maximum Heart Rate:', min_value=0, step=1)
resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'])

# Create a DataFrame with the same column names used in training
input_df = pd.DataFrame({
    'Age': [age],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'MaxHR': [max_hr],
    'RestingECG': [resting_ecg]
})

# Transform the RestingECG feature
input_df['RestingECG'] = label_encoder.transform(input_df['RestingECG'])
input_df_scaled = scaler.transform(input_df)

# Predict using each model
if st.button('Predict Heart Disease Probability'):
    models = {
        'KNN': knn,
        'ANN': ann,
        'SVM': svm
    }

    for model_name, model in models.items():
        prediction = model.predict(input_df_scaled)
        probability = model.predict_proba(input_df_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Display results
        st.write(f"### {model_name} Model")
        st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        if probability is not None:
            st.write(f"Probability of Heart Disease: {probability[0] * 100:.2f}%")
