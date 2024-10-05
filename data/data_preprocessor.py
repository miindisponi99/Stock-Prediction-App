import pandas as pd
import streamlit as st

import logging


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the stock data by handling missing values.
        """
        try:
            self.data.fillna(method='ffill', inplace=True)
            self.data.fillna(method='bfill', inplace=True)
            logging.info("Preprocessed data by handling missing values.")
            return self.data
        except Exception as e:
            st.error(f"Error in preprocessing data: {e}")
            logging.error(f"Error in preprocessing data: {e}")
            return pd.DataFrame()