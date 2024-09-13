import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Load the models and preprocessing steps
heart_failure_model = load('heart_failure_model.joblib')

# Function to make predictions for the unseen data
def predict_unseen_data(models):
    unseen_data = get_unseen_data()
    unseen_df = pd.DataFrame([unseen_data])
    
    # Scale the input data
    unseen_df_scaled = scaler.transform(unseen_df)
    
    for model, model_name in models:
        try:
            # Load the trained model
            model = joblib.load(f'{model_name}.pkl')
            y_pred = model.predict(unseen_df_scaled)
            y_prob = model.predict_proba(unseen_df_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            heart_failure = "Yes" if y_pred[0] == 1 else "No"
            print(f"\n{model_name} Results:")
            print(f"Age Entered: {unseen_data['Age']}")
            if y_prob is not None:
                # Ensure y_prob is a scalar for formatting
                probability = y_prob[0] if isinstance(y_prob, np.ndarray) else y_prob
                print(f"Probability of Heart Disease: {probability * 100:.2f}%")
            else:
                print("Probability information not available")
            print(f"Heart Failure: {heart_failure}")
        except Exception as e:
            print(f"Error with {model_name} model: {e}")

# Define models list for prediction
models = [
    (knn, "KNN"),
    (ann, "ANN"),
    (svm, "SVM")
]

# Call the function to prompt user and predict
predict_unseen_data(models)
