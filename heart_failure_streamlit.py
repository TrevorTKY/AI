import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load the models and preprocessing steps
heart_failure_model = load('heart_failure_model.joblib')

# Extract models and preprocessing steps
knn = heart_failure_model['knn']
ann = heart_failure_model['ann']
svm = heart_failure_model['svm']
scaler = heart_failure_model['scaler']
label_encoder = heart_failure_model['label_encoder']

# Create user input fields
st.title('Heart Disease Prediction')
st.write("Enter the details below to predict the likelihood of heart disease.")

age = st.number_input('Enter age:', min_value=0, max_value=120, step=1)
resting_bp = st.number_input('Resting Blood Pressure:', min_value=0, step=1)
cholesterol = st.number_input('Cholesterol Level:', min_value=0, step=1)
max_hr = st.number_input('Maximum Heart Rate:', min_value=0, step=1)
resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'])

# Create DataFrame for input
input_df = pd.DataFrame({
    'Age': [age],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'MaxHR': [max_hr],
    'RestingECG': [resting_ecg]
})

# Handle unknown labels
if resting_ecg not in label_encoder.classes_:
    st.error(f"Unknown value '{resting_ecg}' for RestingECG. Please select from {label_encoder.classes_}.")
else:
    # Transform and scale input data
    input_df['RestingECG'] = label_encoder.transform(input_df['RestingECG'])
    input_df_scaled = scaler.transform(input_df)

    # Function to make predictions for the unseen data
    def predict_unseen_data(models, input_data):
        st.write("### Prediction Results")

        for model, model_name in models:
            try:
                y_pred = model.predict(input_data)
                y_prob = model.predict_proba(input_data)[:, 1] if hasattr(model, 'predict_proba') else None
                heart_failure = "Yes" if y_pred[0] == 1 else "No"
                
                # Format results
                st.write(f"### {model_name} Results:")
                st.write(f"Age Entered: {age}")
                if y_prob is not None:
                    # Ensure y_prob is a scalar for formatting
                    prob_value = y_prob[0] if isinstance(y_prob, (list, np.ndarray)) else y_prob
                    st.write(f"Probability of Heart Disease: {prob_value * 100:.2f}%")
                else:
                    st.write("Probability information not available")
                st.write(f"Heart Failure: {heart_failure}")
            except Exception as e:
                st.error(f"Error with {model_name} model: {e}")

    # Define models list for prediction
    models = [
        (knn, "KNN"),
        (ann, "ANN"),
        (svm, "SVM")
    ]

    # Call the function to predict
    if st.button('Predict Heart Disease Probability'):
        predict_unseen_data(models, input_df_scaled)
