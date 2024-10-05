import yfinance as yf
import pandas as pd
import streamlit as st

import logging


class DataFetcher:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        """
        try:
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            if data.empty:
                st.error("No data fetched. Please check the ticker and date range.")
                logging.error(f"No data fetched for ticker {self.ticker} between {self.start_date} and {self.end_date}.")
                return pd.DataFrame()
            data.reset_index(inplace=True)
            logging.info(f"Fetched data for {self.ticker} from {self.start_date} to {self.end_date}.")
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            logging.error(f"Error fetching data for ticker {self.ticker}: {e}")
            return pd.DataFrame()