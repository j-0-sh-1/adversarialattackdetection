import streamlit as st
import numpy as np
import joblib

# Load the pre-trained models
heart_disease_model = joblib.load('heart_disease_model.pkl')
log_reg_classifier = joblib.load('log_reg_classifier.pkl')
svm_classifier = joblib.load('svm_classifier.pkl')

def predict_heart_disease(model, patient_data):
    # Code to predict heart disease
    prediction = model.predict(patient_data.reshape(1, -1))
    return "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

def check_data_tampering(model, patient_data):
    # Code to check data tampering
    prediction = model.predict(patient_data.reshape(1, -1))
    return "Data Tampered" if prediction[0] == 1 else "No Data Tampering"

def main():
    st.title("Heart Disease Prediction & Data Tampering Check")

    

    # Create sliders and input fields for patient data
    age = st.number_input('Age', 0, 100, 50)
    age_slider = st.slider('Age Slider', 0, 100, age)

    sex = st.selectbox('Sex', (0, 1), format_func=lambda x: 'Female' if x == 0 else 'Male')

    cp = st.number_input('Chest Pain Type', 0, 50, 1)
    cp_slider = st.slider('Chest Pain Type Slider', 0, 50, cp)

    trestbps = st.number_input('Resting Blood Pressure', 90, 200, 120)
    trestbps_slider = st.slider('Resting Blood Pressure Slider', 90, 200, trestbps)

    chol = st.number_input('Serum Cholestoral in mg/dl', 100, 600, 300)
    chol_slider = st.slider('Serum Cholestoral Slider', 100, 600, chol)

    fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl', 0, 50, 0)
    fbs_slider = st.slider('Fasting Blood Sugar Slider', 0, 50, fbs)

    restecg = st.number_input('Resting Electrocardiographic Results', 0, 50, 1)
    restecg_slider = st.slider('Resting ECG Results Slider', 0, 50, restecg)

    thalach = st.number_input('Maximum Heart Rate Achieved', 70, 220, 130)
    thalach_slider = st.slider('Max Heart Rate Slider', 70, 220, thalach)

    exang = st.number_input('Exercise Induced Angina', 0, 50, 0)
    exang_slider = st.slider('Exercise Induced Angina Slider', 0, 50, exang)

    oldpeak = st.number_input('ST depression induced by exercise', 0.0, 60.0, 2.0)
    oldpeak_slider = st.slider('ST Depression Slider', 0.0, 60.0, oldpeak)

    slope = st.number_input('Slope of the peak exercise ST segment', 0, 10, 1)
    slope_slider = st.slider('Slope Slider', 0, 10, slope)

    ca = st.number_input('Number of major vessels colored by flourosopy', 0, 40, 0)
    ca_slider = st.slider('Major Vessels Slider', 0, 40, ca)

    thal = st.number_input('Thalassemia', 0, 30, 1)
    thal_slider = st.slider('Thalassemia Slider', 0, 30, thal)

    # Aggregate the data into a NumPy array
    #patient_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    # Aggregate the data into a NumPy array using values from the sliders or number inputs
    patient_data = np.array([age_slider, sex, cp_slider, trestbps_slider, chol_slider, fbs_slider, restecg_slider, thalach_slider, exang_slider, oldpeak_slider, slope_slider, ca_slider, thal_slider])


    if st.button('Predict Heart Disease'):
        result = predict_heart_disease(heart_disease_model, patient_data)
        st.write(result)

    tampering_model = st.selectbox('Tampering Check Model', ('Logistic Regression', 'SVM'))

    if st.button('Check Data Tampering'):
        model_to_use = log_reg_classifier if tampering_model == 'Logistic Regression' else svm_classifier
        tampering_result = check_data_tampering(model_to_use, patient_data)
        st.write(tampering_result)

if __name__ == "__main__":
    main()
