import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('Car_price_prediction.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Encode categorical features
    cat_features = ['model', 'motor_type', 'color', 'type', 'status', 'running_km']
    for feature in cat_features:
        data[feature] = data[feature].astype('category').cat.codes
    return data

# Function to predict car prices
def predict_price(data):
    # Preprocess the input data
    processed_data = preprocess_input(data)
    
    # Predict using the trained model
    prediction = model.predict(processed_data)
    
    return prediction

# Main function for Streamlit app
def main():
    # Title of the web app
    st.sidebar.write("<h1 style='color: blue;'>Car Price Prediction</h1>", unsafe_allow_html=True)
    
    # Input fields for features in sidebar
    with st.sidebar:
        st.write("### Enter Car Details")
        model = st.text_input("Model")
        year = st.number_input("Year")
        motor_type = st.text_input("Motor Type")
        color = st.text_input("Color")
        car_type = st.text_input("Car Type")
        status = st.text_input("Status")
        motor_volume = st.number_input("Motor Volume")
        running_km = st.number_input("Running Kilometers")

        if st.button('Predict Price'):
            user_data = pd.DataFrame({
                'model': [model],
                'year': [year],
                'motor_type': [motor_type],
                'color': [color],
                'type': [car_type],
                'status': [status],
                'motor_volume': [motor_volume],
                'running_km': [running_km]
            })

            prediction = predict_price(user_data)

    # Display the predicted price in the center
    st.write("<div style='text-align: center;'><h2>Predicted Price:</h2></div>", unsafe_allow_html=True)
    if 'prediction' in locals():
        st.write(f"<div style='text-align: center;'><h3><font color='red'>${prediction[0]:,.2f}</font></h3></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
