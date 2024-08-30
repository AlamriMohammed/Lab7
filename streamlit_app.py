import streamlit as st
import requests
import json

# Define the URL for your API
API_URL = "https://lab7-ue77.onrender.com/predict"  # Replace with your API endpoint

# Function to get prediction from API
def get_prediction(input_data):
    try:
        # Send a POST request to the API with input data
        response = requests.post(API_URL, json=input_data)
        
        # Handle the response from the API
        if response.status_code == 200:
            prediction = response.json().get("prediction", "No prediction found")
            return prediction
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"


# Streamlit app code
def main():
    st.title("ML Prediction App")

    # Example of form inputs - Adjust based on your model's requirements
    feature1 = st.number_input("Enter Current Value:", value=0)
    feature2 = st.number_input("Enter number of Goals:", value=0)

    # Create a button to make predictions
    if st.button("Predict"):
        # Create input data to send to the API
        input_data = {
            "feature1": feature1,
            "feature2": feature2
        }

        # Get prediction from the API
        prediction = get_prediction(input_data)
        
        # Display the prediction
        st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
