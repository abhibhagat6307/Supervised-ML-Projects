import numpy as np
import pickle
import streamlit as st

# Load model and scaler
model = pickle.load(open("trained_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.set_page_config(page_title="Diabetes Prediction", page_icon="⚕️", layout="centered")

st.markdown("""
    <h2 align="center" style="color: lightblue;">⚕️ Diabetes Prediction System</h2>
    <p align="center" style="color: grey;">Enter your health data below to check your diabetes risk and view key explanations.</p>
""", unsafe_allow_html=True)


with st.form("input_form"):
    c1, c2 = st.columns(2)

    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0)

    with c2:
        insulin = st.number_input("Insulin (µU/mL)", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
        age = st.number_input("Age", min_value=0)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted  and age > 10:
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]).reshape(1, -1)
    scaled_data = scaler.transform(input_data)

    result = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1] * 100

    
    explanation = {}

    if glucose > 140:
        explanation["Glucose Level"] = f"High glucose levels ({glucose} mg/dL) indicate a higher risk of diabetes."
    else:
        explanation["Glucose Level"] = f"Glucose levels ({glucose} mg/dL) are within normal range."

    if bmi > 30:
        explanation["BMI"] = f"A BMI of {bmi} indicates overweight, which increases diabetes risk."
    else:
        explanation["BMI"] = f"BMI of {bmi} is within a healthy range."

    if insulin > 150:
        explanation["Insulin Level"] = f"Insulin levels ({insulin} µU/mL) are high, suggesting possible insulin resistance."
    else:
        explanation["Insulin Level"] = f"Insulin levels ({insulin} µU/mL) are within normal range."

    if age > 45:
        explanation["Age"] = f"Age {age} is a significant risk factor for diabetes."
    else:
        explanation["Age"] = f"Age {age} is not a major risk factor for diabetes."


    st.markdown("---")
    st.subheader("Prediction Result")

    if result == 1:
        st.success(f"Diabetes Suspected — Risk Probability: **{probability:.2f}%**")
    else:
        st.error(f"Diabetes Not Suspected — Risk Probability: **{probability:.2f}%**")

    # Show explanations
    st.markdown("### Health Factor Analysis:")
    for key, value in explanation.items():
        st.write(f"- **{key}:** {value}")

    # Recommendations
    st.markdown("### Recommendations:")
    if result == 1:
        st.write("- Consult a doctor for further glucose testing.")
        st.write("- Maintain a low-sugar, high-fiber diet.")
        st.write("- Include regular exercise in your daily routine.")
    else:
        st.write("- Maintain a balanced diet and regular health check-ups.")
        st.write("- Continue healthy habits to stay diabetes-free.")