import streamlit as st
import pandas as pd
from joblib import load

# Load the models and preprocessing steps
heart_failure_model = load('heart_failure_model.joblib')

# Extract models and preprocessing steps
knn = heart_failure_model['knn']
scaler = heart_failure_model['scaler']
label_encoder = heart_failure_model['label_encoder']
poly = heart_failure_model['poly']
feature_selector = heart_failure_model['feature_selector']

# Set the Streamlit page configuration
st.set_page_config(page_title='Heart Failure Prediction', page_icon=':heart:', layout='wide')

# Title and description
st.title('Heart Failure Prediction System')
st.write("Enter the details below to predict the likelihood of heart failure using the KNN model.")

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
    input_df_poly = poly.transform(input_df_scaled)
    input_df_selected = feature_selector.transform(input_df_poly)

    # Prediction using KNN model
    if st.button('Predict Heart Failure'):
        y_pred = knn.predict(input_df_selected)
        y_prob = knn.predict_proba(input_df_selected)[:, 1]  # Get the probability of heart failure

        # Display results
        probability = y_prob[0] * 100  # Convert to percentage
        heart_failure = "Yes" if y_pred[0] == 1 else "No"

        # Display entered details
        st.write(f"### Entered Details")
        st.write(f"- *Age:* {age}")
        st.write(f"- *Resting Blood Pressure:* {resting_bp} mm Hg")
        st.write(f"- *Cholesterol Level:* {cholesterol} mg/dl")
        st.write(f"- *Maximum Heart Rate:* {max_hr} bpm")
        st.write(f"- *Resting ECG:* {resting_ecg}")

        # Display prediction result
        st.write(f"### Prediction Results")
        st.write(f"- *Heart Failure Likelihood:* {heart_failure}")
        st.write(f"- *Predicted Probability of Heart Failure:* {probability:.2f}%")

        # Additional message based on prediction
        if y_pred[0] == 1:
            st.warning("Based on the prediction, there is a significant chance that you might develop heart failure.")
        else:
            st.success("Based on the prediction, it is unlikely that you will develop heart failure.")

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
