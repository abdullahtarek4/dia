import streamlit as st
import numpy as np
import joblib

# Load your trained diabetes prediction model
model = joblib.load(open(r'best_model.pkl', 'rb'))

# Define features based on your model training
features = ['Age', 'BMI', 'Physical_Activity_Level', 'Smoking_Status',
       'Alcohol_Consumption', 'Family_History_of_Diabetes',
       'Fasting_Blood_Glucose', 'HbA1c', 'Blood_Pressure_Systolic',
       'Blood_Pressure_Diastolic']


def main():
    st.set_page_config(layout='wide')
    st.title('Diabetes Prediction App')

    inputs = {}
    for feature in features:
        # Numeric inputs for diabetes prediction
        inputs[feature] = st.number_input(f'Enter {feature}:', step=0.1)

    if st.button('Predict'):
        input_values = np.array([inputs[feature] for feature in features]).reshape(1, -1)
        prediction = model.predict(input_values)

        if prediction[0] == 1:
            st.error("⚠️ The model predicts a high risk of Diabetes. Please consult a doctor.")
        else:
            st.success("✅ The model predicts a low risk of Diabetes. Keep maintaining a healthy lifestyle.")

if __name__ == '__main__':
    main()
