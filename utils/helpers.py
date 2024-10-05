import streamlit as st
import matplotlib.pyplot as plt


def plot_predictions(dates, actual, predicted):
    """
    Plot actual vs predicted stock prices.
    """
    plt.figure(figsize=(14,7))
    plt.plot(dates, actual, label='Actual Price')
    plt.plot(dates, predicted, label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)