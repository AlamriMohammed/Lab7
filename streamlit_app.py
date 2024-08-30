import streamlit as st
import requests

# Image file path (Make sure the image is in the same directory as your script or provide the full path)
image_path = "pic.png"  # Replace with your actual image file name or path

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


    # Input for current_value, kept as an integer
    feature1 = st.number_input("Enter Current Value:", value=0)

    # Input for goals, changed to a float with a range of 0 to 1
    feature2 = st.number_input("Enter number of Goals (0 to 1):", value=0.0, min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict"):
        input_data = {
            "current_value": feature1,
            "goals": feature2
        }

        prediction = get_prediction(input_data)
        st.write(f"Prediction: {prediction}")
    st.image(image_path, caption="Visual Representation", use_column_width=True)


if __name__ == '__main__':
    main()
