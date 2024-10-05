import streamlit as st
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


class ModelFactory:
    @staticmethod
    def get_model(model_name: str, params: dict = None):
        """
        Factory method to get the appropriate model based on the name.

        Parameters:
        - model_name: Name of the machine learning model.
        - params: Dictionary of hyperparameters for the model.

        Returns:
        - An instance of the requested machine learning model with the specified hyperparameters.
        """
        if params is None:
            params = {}
        
        try:
            if model_name == "Random Forest":
                return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == "XGBoost":
                return XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
            elif model_name == "LightGBM":
                return lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == "SVR":
                return SVR(**params)
            elif model_name == "KNN":
                return KNeighborsRegressor(**params, n_jobs=-1)
            elif model_name == "ElasticNet":
                return ElasticNet(**params, random_state=42, max_iter=10000)
            elif model_name == "Decision Tree":
                return DecisionTreeRegressor(**params, random_state=42)
            elif model_name == "Ridge":
                return Ridge(**params, random_state=42)
            elif model_name == "Lasso":
                return Lasso(**params, random_state=42, max_iter=10000)
            elif model_name == "SGDRegressor":
                return SGDRegressor(**params, random_state=42, max_iter=1000, tol=1e-3)
            elif model_name == "CatBoost":
                return CatBoostRegressor(**params, random_state=42, verbose=0)
            elif model_name == "Gaussian Process":
                return GaussianProcessRegressor(**params, random_state=42)
            elif model_name == "MLPRegressor":
                return MLPRegressor(**params, random_state=42, max_iter=1000)
            else:
                st.error(f"Unsupported model selected: {model_name}")
                logging.error(f"Unsupported model selected: {model_name}")
                return None
        except TypeError as te:
            st.error(f"Error initializing {model_name}: {te}")
            logging.error(f"Error initializing {model_name}: {te}")
            return None
        except Exception as e:
            st.error(f"Unexpected error initializing {model_name}: {e}")
            logging.error(f"Unexpected error initializing {model_name}: {e}")
            return None