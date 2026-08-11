
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model_data = joblib.load("student_model.pkl")

if isinstance(model_data, dict):
    model = model_data["model"]
    features = model_data["features"]
    encoders = model_data["encoders"]
else:
    model = model_data
    features = list(model.feature_names_in_)
    encoders = {}

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result {
    text-align: center;
    padding: 30px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">🎓 Student Grade Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered student final grade prediction system</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Student Information
# --------------------------------------------------

st.subheader("📋 Student Information")

st.info(
    "Enter the student details below. "
    "Only features used by the trained model are required."
)

# --------------------------------------------------
# Create Input Dictionary
# --------------------------------------------------

input_values = {}

# Numeric columns from the dataset
numeric_ranges = {
    "age": (15, 25, 17),
    "Medu": (0, 4, 2),
    "Fedu": (0, 4, 2),
    "traveltime": (1, 4, 1),
    "studytime": (1, 4, 2),
    "failures": (0, 4, 0),
    "famrel": (1, 5, 4),
    "freetime": (1, 5, 3),
    "goout": (1, 5, 3),
    "Dalc": (1, 5, 1),
    "Walc": (1, 5, 1),
    "health": (1, 5, 3),
    "absences": (0, 100, 5),
    "G1": (0, 20, 10),
    "G2": (0, 20, 10)
}

# --------------------------------------------------
# Input Fields
# --------------------------------------------------

col1, col2 = st.columns(2)

for index, feature in enumerate(features):

    # Categorical feature
    if feature in encoders:

        encoder = encoders[feature]

        options = list(encoder.classes_)

        if index % 2 == 0:
            with col1:
                value = st.selectbox(
                    feature,
                    options,
                    key=feature
                )
        else:
            with col2:
                value = st.selectbox(
                    feature,
                    options,
                    key=feature
                )

        input_values[feature] = encoder.transform([value])[0]

    # Numeric feature
    else:

        if feature in numeric_ranges:

            min_value, max_value, default_value = numeric_ranges[feature]

        else:

            min_value = 0
            max_value = 100
            default_value = 0

        if index % 2 == 0:
            with col1:
                value = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    key=feature
                )
        else:
            with col2:
                value = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    key=feature
                )

        input_values[feature] = value

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Final Grade",
    use_container_width=True
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    try:

        # Create dataframe with EXACT model features
        input_data = pd.DataFrame([input_values])

        # Make sure feature order is exactly the same
        input_data = input_data[features]

        # Prediction
        prediction = model.predict(input_data)[0]

        # Result
        st.divider()

        st.markdown(
            '<div class="result">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Prediction Result")

        st.markdown(
            f"<h1>{prediction}</h1>",
            unsafe_allow_html=True
        )

        st.write("Predicted Final Grade")

        st.markdown("</div>", unsafe_allow_html=True)

        # Performance message

        if prediction >= 15:

            st.success(
                "🌟 Excellent Performance!"
            )

        elif prediction >= 10:

            st.info(
                "👍 Good / Average Performance"
            )

        else:

            st.warning(
                "⚠️ Student may need additional academic support."
            )

        # Show input data

        with st.expander("🔍 View Input Data"):

            st.dataframe(input_data)

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.exception(e)
