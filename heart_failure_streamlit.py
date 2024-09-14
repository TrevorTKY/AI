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
