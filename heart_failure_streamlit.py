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
st.set_page_config(page_title='Heart Failure Prediction', page_icon=':heart:', layout='wide')

st.title('Heart Failure Prediction System')
st.write("Enter the details below to predict the likelihood of heart failure.")

age = st.number_input('Age:', min_value=0, max_value=120, step=1, help="Enter the age of the patient.")
resting_bp = st.number_input('Resting Blood Pressure (mm Hg):', min_value=0, step=1, help="Enter the resting blood pressure.")
cholesterol = st.number_input('Cholesterol Level (mg/dl):', min_value=0, step=1, help="Enter the cholesterol level.")
max_hr = st.number_input('Maximum Heart Rate (bpm):', min_value=0, step=1, help="Enter the maximum heart rate.")
resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'], help="Select the type of resting ECG.")

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
        st.markdown("#### Model Performance")
        results = []
        
        for model, model_name in models:
            try:
                y_pred = model.predict(input_data)
                y_prob = model.predict_proba(input_data)[:, 1] if hasattr(model, 'predict_proba') else None
                heart_failure = "Yes" if y_pred[0] == 1 else "No"
                
                # Format results
                result = {
                    "Model": model_name,
                    "Age Entered": age,
                    "Probability of Heart Disease": f"{y_prob[0] * 100:.2f}%" if y_prob is not None else "N/A",
                    "Heart Failure": heart_failure
                }
                results.append(result)

            except Exception as e:
                st.error(f"Error with {model_name} model: {e}")

        # Display results in a table
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Additional visualization or details can be added here if needed

    # Define models list for prediction
    models = [
        (knn, "KNN"),
        (ann, "ANN"),
        (svm, "SVM")
    ]

    # Call the function to predict
    if st.button('Predict Heart Failure'):
        predict_unseen_data(models, input_df_scaled)

# Customizing Streamlit theme via markdown
st.markdown("""
    <style>
    .css-1n76uvr {
        background-color: #f0f2f6; /* Light background color */
    }
    .css-1n76uvr .css-1g1z3l2 { /* Adjusts the sidebar color */
        background-color: #003366; /* Dark blue sidebar */
        color: white;
    }
    .css-1n76uvr .css-12yzwg4 { /* Adjusts button color */
        background-color: #007bff; /* Bootstrap primary blue */
        color: white;
    }
    .css-1n76uvr .css-12yzwg4:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    .css-1n76uvr .css-1v0t7x4 { /* Adjusts text color */
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)
