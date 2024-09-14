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

# Set up page configuration and custom styles
st.set_page_config(page_title='Heart Failure Prediction', page_icon=':heart:', layout='wide')

# Custom CSS for background and form styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #5DADE2, #1F618D); /* Blue gradient */
        color: #fff;
    }
    .stButton button {
        background-color: #1A5276;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #154360;
    }
    .stTextInput label {
        color: #fff;  /* Label color */
    }
    .stSelectbox label {
        color: #fff; /* Label color for selectbox */
    }
    .css-10trblm {
        background-color: #154360; /* Box background color */
        border: none;
    }
    .css-12yzwg4 {
        background-color: #1A5276;
        border-radius: 8px;
    }
    .css-12yzwg4:hover {
        background-color: #154360; /* Darker hover */
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title('Heart Failure Prediction System')
st.write("Enter the details below to predict the likelihood of heart failure.")

# Create input fields
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

        # Plotting the probabilities
        st.write("### Probability of Heart Disease for Each Model")
        fig, ax = plt.subplots(figsize=(10, 6))
        models_names = [result['Model'] for result in results]
        probabilities = [float(result['Probability of Heart Disease'].replace('%', '')) for result in results]
        
        ax.bar(models_names, probabilities, color=['#2980B9', '#27AE60', '#E74C3C'])
        ax.set_xlabel('Model')
        ax.set_ylabel('Probability (%)')
        ax.set_title('Probability of Heart Disease for Each Model')
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig)

    # Define models list for prediction
    models = [
        (knn, "KNN"),
        (ann, "ANN"),
        (svm, "SVM")
    ]

    # Call the function to predict
    if st.button('Predict Heart Failure'):
        predict_unseen_data(models, input_df_scaled)
