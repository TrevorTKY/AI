import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Load the models and preprocessing steps
heart_failure_model = load('heart_failure_model.joblib')

# Extract models and preprocessing steps
knn = heart_failure_model['knn']
ann = heart_failure_model['ann']
svm = heart_failure_model['svm']
scaler = heart_failure_model['scaler']
label_encoder = heart_failure_model['label_encoder']

# Set the Streamlit page configuration
st.set_page_config(page_title='Heart Failure Prediction', page_icon=':heart:', layout='wide')

# Title and description
st.title('Heart Failure Prediction System')
st.write("Enter the details below to predict the likelihood of heart failure based on various models (KNN, ANN, SVM).")

# Input fields for user data
age = st.number_input('Age:', min_value=0, max_value=120, step=1, help="Enter the age of the patient.")
resting_bp = st.number_input('Resting Blood Pressure (mm Hg):', min_value=0, step=1, help="Enter the resting blood pressure.")
cholesterol = st.number_input('Cholesterol Level (mg/dl):', min_value=0, step=1, help="Enter the cholesterol level.")
max_hr = st.number_input('Maximum Heart Rate (bpm):', min_value=0, step=1, help="Enter the maximum heart rate.")
resting_ecg = st.selectbox('Resting ECG:', ['Normal', 'ST', 'LVH'], help="Select the type of resting ECG.")

# Create a DataFrame for the input data
input_df = pd.DataFrame({
    'Age': [age],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'MaxHR': [max_hr],
    'RestingECG': [resting_ecg]
})

# Handle unknown labels for RestingECG
if resting_ecg not in label_encoder.classes_:
    st.error(f"Unknown value '{resting_ecg}' for RestingECG. Please select from {label_encoder.classes_}.")
else:
    # Transform and scale input data
    input_df['RestingECG'] = label_encoder.transform(input_df['RestingECG'])
    input_df_scaled = scaler.transform(input_df)

    # Function to make predictions using multiple models and highlight the best model
    def predict_unseen_data(models, input_data):
        st.write("### Prediction Results")
        best_model = None
        best_model_name = None
        best_probability = 0
        
        results = []
        
        for model, model_name in models:
            try:
                y_pred = model.predict(input_data)
                y_prob = model.predict_proba(input_data)[:, 1] if hasattr(model, 'predict_proba') else None
                heart_failure = "Yes" if y_pred[0] == 1 else "No"
                
                # Determine the best model based on probability
                if y_prob is not None and y_prob[0] > best_probability:
                    best_probability = y_prob[0]
                    best_model = model
                    best_model_name = model_name
                    
                result = {
                    "Model": model_name,
                    "Age Entered": age,
                    "Probability of Heart Disease": f"{y_prob[0] * 100:.2f}%" if y_prob is not None else "N/A",
                    "Heart Failure": heart_failure
                }
                results.append(result)

            except Exception as e:
                st.error(f"Error with {model_name} model: {e}")

        # Display the result for the best model
        if best_model:
            best_result = next(result for result in results if result['Model'] == best_model_name)
            st.write(f"**Best Model: {best_model_name}**")
            st.write(pd.DataFrame([best_result]))

            # Plotting the probability for the best model
            st.write("### Probability of Heart Disease for the Best Model")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(best_model_name, float(best_result['Probability of Heart Disease'].replace('%', '')), color='#007bff')
            ax.set_xlabel('Model')
            ax.set_ylabel('Probability (%)')
            ax.set_title('Probability of Heart Disease for the Best Model')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')

            st.pyplot(fig)
        else:
            st.write("No model was able to make a prediction.")

    # List of models for prediction
    models = [
        (knn, "KNN"),
        (ann, "ANN"),
        (svm, "SVM")
    ]

    # Trigger prediction when the button is clicked
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
