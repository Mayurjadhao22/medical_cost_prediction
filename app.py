import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. Load the trained XGBoost model
# Change 'model.pkl' to whatever filename your model is saved as
MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# 2. App UI Configuration
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("Medical Insurance Cost Predictor")
st.write("Enter the details below to estimate annual medical insurance charges.")

# 3. User Input Fields
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

# 4. Encode categorical inputs to match the model's expected integer types
# Standard LabelEncoder alphabetical mapping:
sex_map = {"Female": 0, "Male": 1}
smoker_map = {"No": 0, "Yes": 1}
region_map = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

# 5. Predict Button
if st.button("Predict Insurance Cost", type="primary"):
    # Create DataFrame with exact feature names and order expected by the XGBoost model
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
