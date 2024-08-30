import streamlit as st
import requests
import json

API_URL = "https://lab7-ue77.onrender.com/predict"  # Replace with your API endpoint

def get_prediction(input_data):
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json().get("cluster", "No cluster found")
            return prediction
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def main():
    st.title("ML Prediction App")

    feature1 = st.number_input("Enter Current Value:", value=0)
    feature2 = st.number_input("Enter number of Goals:", value=0)

    if st.button("Predict"):
        input_data = {
            "current_value": feature1,
            "goals": feature2
        }

        prediction = get_prediction(input_data)
        st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
