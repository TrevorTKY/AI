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
st.title('Heart Failure Prediction')
st.write("Please enter the details below to predict the likelihood of heart failure. The models used for prediction are KNN, ANN, and SVM.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age:', min_value=0, max_value=120, step=1)
    resting_bp = st.number_input('Resting Blood Pressure (mm Hg):', min_value=0, step=1)
    cholesterol = st.number_input('Cholesterol Level (mg/dL):', min_value=0, step=1)

with col2:
    max_hr = st.number_input('Maximum Heart Rate (bpm):', min_value=0, step=1)
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

        results = []
        for model, model_name in models:
            try:
                y_pred = model.predict(input_data)
                y_prob = model.predict_proba(input_data)[:, 1] if hasattr(model, 'predict_proba') else None
                heart_failure = "Yes" if y_pred[0] == 1 else "No"
                
                # Format results
                result = {
                    'Model': model_name,
                    'Prediction': heart_failure,
                    'Probability': f"{y_prob[0] * 100:.2f}%" if y_prob is not None else "N/A"
                }
                results.append(result)
            except Exception as e:
                st.error(f"Error with {model_name} model: {e}")

        # Display results in a more structured format
        results_df = pd.DataFrame(results)
        st.table(results_df)

        # Plot results
        st.bar_chart(results_df.set_index('Model')['Probability'].apply(lambda x: float(x.rstrip('%'))))

    # Define models list for prediction
    models = [
        (knn, "KNN"),
        (ann, "ANN"),
        (svm, "SVM")
    ]

    # Call the function to predict
    if st.button('Predict Heart Failure'):
        with st.spinner('Making predictions...'):
            predict_unseen_data(models, input_df_scaled)
