import numpy as np
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt

from data_loader import load_data
from model import train_model

st.title("Stock Price Prediction App")

stock = st.text_input("Enter Stock Symbol", "AAPL")

data = load_data(stock)

st.subheader("Raw Data")
st.write(data.tail())

#Plot closing price 
st.subheader("Closing Price Chart")
fig = plt.figure()
plt.plot(data['Close'])
st.pyplot(fig)

# Train model 
model, scaler = train_model(data[['Close']])

# Predict next day
last_60_days = data['Close'].tail(60).values
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1,1))

X_test = []
X_test.append(last_60_days_scaled)


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)

st.subheader(f"Predicted Next Price: {pred_price[0][0]}")