import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("housing.csv")

# Fill missing values
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("AI-Based Real Estate Valuation System")

st.write("Predict house prices using Machine Learning")

# Sidebar inputs
st.sidebar.header("Enter House Details")

housing_age = st.sidebar.number_input("Housing Median Age", 0)

total_rooms = st.sidebar.number_input("Total Rooms", 0)

total_bedrooms = st.sidebar.number_input("Total Bedrooms", 0)

population = st.sidebar.number_input("Population", 0)

households = st.sidebar.number_input("Households", 0)

median_income = st.sidebar.number_input("Median Income", 0.0)

# Prediction button
if st.sidebar.button("Predict Price"):

    features = np.array([[housing_age,
                          total_rooms,
                          total_bedrooms,
                          population,
                          households,
                          median_income]])

    prediction = model.predict(features)

    # Convert prediction to USD first
    usd_price = prediction[0]

    # Convert USD to INR
    inr_price = usd_price * 83

    st.success(f"Predicted House Price: ₹{inr_price:,.2f}")
# -------- DATA VISUALIZATION --------

st.subheader("House Price Distribution")

fig1 = plt.figure()
sns.histplot(df['median_house_value'], bins=50)
plt.xlabel("House Price")
plt.ylabel("Count")

st.pyplot(fig1)

# -------- CORRELATION HEATMAP --------

st.subheader("Feature Correlation")

fig2 = plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")

st.pyplot(fig2)

# -------- MAP VISUALIZATION --------

st.subheader("Housing Locations")

map_data = df[['latitude','longitude']]

st.map(map_data)