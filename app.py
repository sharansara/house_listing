import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the encoder object and the best-performing model
encoder = pickle.load(open("encoder.pkl", "rb"))
best_model = pickle.load(open("best_model.pkl", "rb"))

# Load the dataset
data = pd.read_csv("HouseListings-Top45Cities-10292023-kaggle.csv", encoding='ISO-8859-1')

# Define categorical and numerical features
categorical_features = ['City', 'Province']
numerical_features = ['Price', 'Number_Beds', 'Number_Baths', 'Population', 'Median_Family_Income', 'Price_per_Bedroom']

# Function to preprocess input data
def preprocess_input(input_data):
    # Encode categorical features
    encoded_features = encoder.transform(input_data[categorical_features])
    # Scale numerical features
    numerical_data = input_data[numerical_features]
    scaled_numerical_data = (numerical_data - numerical_data.mean()) / numerical_data.std()  # Standardization
    # Combine encoded and scaled numerical features
    processed_input = np.hstack((encoded_features.toarray(), scaled_numerical_data))
    return processed_input

# Function to predict house prices
def predict_price(input_data):
    processed_input = preprocess_input(input_data)
    predicted_price = best_model.predict(processed_input.reshape(1, -1))
    return predicted_price[0]

# Create a Streamlit UI
st.title("House Price Prediction")

# Collect input features from the user
city = st.selectbox("City", data['City'].unique())
province = st.selectbox("Province", data['Province'].unique())
price = st.number_input("Price", value=0)
number_beds = st.number_input("Number of Bedrooms", value=1)
number_baths = st.number_input("Number of Bathrooms", value=1)
population = st.number_input("Population", value=0)
median_family_income = st.number_input("Median Family Income", value=0)
price_per_bedroom = price / number_beds if number_beds != 0 else 0

# Create a dictionary with user input
user_input = {
    'City': city,
    'Province': province,
    'Price': price,
    'Number_Beds': number_beds,
    'Number_Baths': number_baths,
    'Population': population,
    'Median_Family_Income': median_family_income,
    'Price_per_Bedroom': price_per_bedroom
}

# Predict house price when the user clicks the button
if st.button("Predict"):
    # Convert the user input into a DataFrame
    input_df = pd.DataFrame([user_input])
    # Predict the house price
    predicted_price = predict_price(input_df)
    # Display the predicted price
    st.success(f"Predicted House Price: ${predicted_price:.2f}")
