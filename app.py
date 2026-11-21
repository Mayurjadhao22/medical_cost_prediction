import glob
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. App UI Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")

# 2. Locate and load the model reliably on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    # Primary expected filename
    default_path = os.path.join(BASE_DIR, "model.pkl")
    if os.path.exists(default_path):
        return joblib.load(default_path)

    # Auto-detect if you named the file differently in your GitHub repo
    for ext in ("*.pkl", "*.joblib", "*.sav", "*.bin", "*.model"):
        matches = glob.glob(os.path.join(BASE_DIR, ext))
        if matches:
            return joblib.load(matches[0])

    return None


model = load_model()

if model is None:
    st.error("Model file not found in the repository.")
    st.info(
        f"Files currently found next to app.py: `{os.listdir(BASE_DIR)}`\n\n"
        "Please upload your trained model file (e.g., `model.pkl`) to the root folder "
        "of your `medical_cost_prediction` GitHub repository and reboot the app."
    )
    st.stop()

# 3. Main App Interface
st.title("Medical Insurance Cost Predictor")
st.write("Enter the details below to estimate annual medical insurance charges.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input(
        "Number of Children", min_value=0, max_value=10, value=0, step=1
    )

with col2:
    sex = st.selectbox("Sex", options=["Female", "Male"])
    smoker = st.selectbox("Smoker", options=["No", "Yes"])
    region = st.selectbox(
        "Region", options=["Northeast", "Northwest", "Southeast", "Southwest"]
    )

# 4. Encode categorical inputs
sex_map = {"Female": 0, "Male": 1}
smoker_map = {"No": 0, "Yes": 1}
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

# 5. Predict Button
if st.button("Predict Insurance Cost", type="primary"):
    input_data = pd.DataFrame(
        [
            {
                "age": int(age),
                "sex": int(sex_map[sex]),
                "bmi": float(bmi),
                "children": int(children),
                "smoker": int(smoker_map[smoker]),
                "region": int(region_map[region]),
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"### Estimated Annual Insurance Cost: ${prediction:,.2f}")
