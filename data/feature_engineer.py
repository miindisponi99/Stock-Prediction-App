import pandas as pd
import streamlit as st

import ta
import logging


class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def engineer_features(self) -> pd.DataFrame:
        """
        Create technical indicators and additional features.
        """
        try:
            self.data['Close_Lag1'] = self.data['Adj Close'].shift(1)
            self.data['Close_Lag2'] = self.data['Adj Close'].shift(2)
            self.data['MA_10'] = ta.trend.SMAIndicator(close=self.data['Adj Close'], window=10).sma_indicator()
            self.data['MA_50'] = ta.trend.SMAIndicator(close=self.data['Adj Close'], window=50).sma_indicator()
            self.data['RSI'] = ta.momentum.RSIIndicator(close=self.data['Adj Close'], window=14).rsi()
            macd = ta.trend.MACD(close=self.data['Adj Close'])
            self.data['MACD'] = macd.macd()
            self.data['MACD_Signal'] = macd.macd_signal()
            self.data['MACD_Diff'] = macd.macd_diff()
            bollinger = ta.volatility.BollingerBands(close=self.data['Adj Close'], window=20, window_dev=2)
            self.data['Bollinger_High'] = bollinger.bollinger_hband()
            self.data['Bollinger_Low'] = bollinger.bollinger_lband()
            self.data['Bollinger_Width'] = bollinger.bollinger_wband()
            self.data['Daily_Return'] = self.data['Adj Close'].pct_change()
            self.data['Day_of_Week'] = self.data['Date'].dt.dayofweek
            self.data['Month'] = self.data['Date'].dt.month
            self.data['Target'] = self.data['Adj Close'].shift(-1)
            self.data.dropna(inplace=True)
            logging.info("Engineered features including lagged prices, technical indicators, and target variable.")
            return self.data
        except Exception as e:
            st.error(f"Error in feature engineering: {e}")
            logging.error(f"Error in feature engineering: {e}")
            return pd.DataFrame()