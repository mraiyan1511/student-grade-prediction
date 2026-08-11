import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Grade Prediction")
st.write("Enter the student details below to predict the final grade.")

st.divider()

# Student inputs
st.subheader("📚 Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=15, max_value=25, value=17)
    studytime = st.number_input("Study Time", min_value=1, max_value=4, value=2)
    failures = st.number_input("Past Failures", min_value=0, max_value=4, value=0)
    absences = st.number_input("Absences", min_value=0, max_value=100, value=5)

with col2:
    G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10)
    G2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10)
    Medu = st.number_input("Mother's Education", min_value=0, max_value=4, value=2)
    Fedu = st.number_input("Father's Education", min_value=0, max_value=4, value=2)

st.divider()

# Prediction
if st.button("🔮 Predict Grade", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G1": [G1],
        "G2": [G2],
        "Medu": [Medu],
        "Fedu": [Fedu]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.success(f"🎯 Predicted Final Grade: {prediction}")

        if prediction >= 15:
            st.info("🌟 Excellent Performance")
        elif prediction >= 10:
            st.info("👍 Good / Average Performance")
        else:
            st.warning("⚠️ Student may need additional support.")

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)
