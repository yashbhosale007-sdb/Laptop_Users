import streamlit as st
import pandas as pd
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open("model1.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Classification Prediction App")
st.write("Enter the details below to get a prediction from the model.")

# Create input fields based on the model's feature names 
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    region = st.text_input("Region", value="North")

with col2:
    occupation = st.text_input("Occupation", value="Professional")
    income = st.number_input("Income", min_value=0, value=50000)

# Create a DataFrame for the prediction
# Note: Ensure categorical variables (Gender, Region, Occupation) match the 
# encoding used during model training (e.g., Label Encoding or One-Hot).
input_data = pd.DataFrame([[age, gender, region, occupation, income]], 
                          columns=['Age', 'Gender', 'Region', 'Occupation', 'Income'])

if st.button("Predict"):
    try:
        # Perform prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        st.divider()
        st.subheader(f"Result: {prediction[0]}")
        
        # Show confidence levels
        st.write(f"Confidence: {max(probability[0]) * 100:.2f}%")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Check if your categorical inputs (Gender, Region, Occupation) need specific numerical encoding.")
