import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the trained model
with open("house_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the California Housing Dataset
california_housing = fetch_california_housing()
house_price_dataframe = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
house_price_dataframe['price'] = california_housing.target

# Streamlit UI Setup
st.title("California House Price Prediction App 🏡")
st.write("Enter the house details to predict the price in Indian Rupees.")

# User Input Fields
col1, col2 = st.columns(2)
with col1:
    MedInc = st.number_input("Median Income (in 10k USD)", min_value=0.0, step=0.1)
    HouseAge = st.number_input("House Age (years)", min_value=0, step=1)
    AveRooms = st.number_input("Average Rooms per Household", min_value=0.0, step=0.1)
    AveBedrms = st.number_input("Average Bedrooms per Household", min_value=0.0, step=0.1)
with col2:
    Population = st.number_input("Population", min_value=0, step=1)
    AveOccup = st.number_input("Average Occupancy per Household", min_value=0.0, step=0.1)
    Latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.1)
    Longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.1)

# Predict Button
if st.button("Predict Price 💰"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)[0] * 100000  # Converting to INR
    st.success(f"Estimated House Price: ₹{prediction:,.2f}")

    # Visualization Section
    st.subheader("Data Insights and Visualizations 📊")
    
    # 1. Distribution of House Prices
    fig, ax = plt.subplots()
    sns.histplot(house_price_dataframe['price'] * 100000, bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of House Prices (in INR)")
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    st.write("This histogram shows the distribution of house prices, indicating the most common price ranges.")

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(house_price_dataframe.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap of Features")
    st.pyplot(fig)
    st.write("This heatmap displays the relationships between different housing features and the target price.")

    # 3. House Price vs. Median Income
    fig, ax = plt.subplots()
    sns.scatterplot(x=house_price_dataframe['MedInc'], y=house_price_dataframe['price'], alpha=0.5, ax=ax)
    ax.set_title("House Price vs. Median Income")
    ax.set_xlabel("Median Income (in 10k USD)")
    ax.set_ylabel("Price (in 100k USD)")
    st.pyplot(fig)
    st.write("This scatter plot shows that higher median incomes generally correspond to higher house prices.")

    # 4. House Price vs. House Age
    fig, ax = plt.subplots()
    sns.boxplot(x=house_price_dataframe['HouseAge'], y=house_price_dataframe['price'], ax=ax)
    ax.set_title("House Price Distribution by Age")
    ax.set_xlabel("House Age (years)")
    ax.set_ylabel("Price (in 100k USD)")
    st.pyplot(fig)
    st.write("This box plot illustrates how house prices vary based on the age of the house.")

    # 5. Population vs. House Price
    fig, ax = plt.subplots()
    sns.scatterplot(x=house_price_dataframe['Population'], y=house_price_dataframe['price'], alpha=0.5, ax=ax)
    ax.set_title("Population vs. House Price")
    ax.set_xlabel("Population")
    ax.set_ylabel("Price (in 100k USD)")
    st.pyplot(fig)
    st.write("This scatter plot shows how population density in an area influences house prices.")
