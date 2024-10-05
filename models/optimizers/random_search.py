import numpy as np
import streamlit as st

import logging

from sklearn.model_selection import RandomizedSearchCV
from models.model_factory import ModelFactory
from scipy.stats import randint, uniform
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


class RandomSearchOptimizer:
    def __init__(self, model_name: str, X_train, y_train):
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train

    def get_param_distributions(self):
        """
        Define parameter distributions based on the model name.
        """
        if self.model_name == "Random Forest":
            return {
                'n_estimators': [int(x) for x in np.linspace(start=50, stop=300, num=6)],
                'max_depth': [int(x) for x in np.linspace(5, 50, num=10)] + [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_name == "XGBoost":
            return {
                'n_estimators': [int(x) for x in np.linspace(start=50, stop=300, num=6)],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
        elif self.model_name == "LightGBM":
            return {
                'n_estimators': [int(x) for x in np.linspace(start=50, stop=300, num=6)],
                'max_depth': [10, 20, 30],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        elif self.model_name == "SVR":
            return {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5, 1.0],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        elif self.model_name == "KNN":
            return {
                'n_neighbors': list(range(1, 31)),
                'weights': ['uniform', 'distance']
            }
        elif self.model_name == "ElasticNet":
            return {
                'alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
                'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
            }
        elif self.model_name == "Decision Tree":
            return {
                'max_depth': [10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_name == "Ridge":
            return {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
            }
        elif self.model_name == "Lasso":
            return {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'selection': ['cyclic', 'random']
            }
        elif self.model_name == "SGDRegressor":
            return {
                'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
            }
        elif self.model_name == "CatBoost":
            return {
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'iterations': [100, 300, 500, 700, 1000],
                'l2_leaf_reg': [1, 3, 5, 7, 10]
            }
        elif self.model_name == "Gaussian Process":
            return {
                'kernel': [
                    ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
                    ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e2))
                ],
                'alpha': [1e-10, 1e-5, 1e-3, 1e-2]
            }
        elif self.model_name == "MLPRegressor":
            return {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'solver': ['adam', 'lbfgs', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        else:
            st.error("Unsupported model selected for Random Search.")
            return None

    def optimize(self, n_iter=20):
        """
        Perform Random Search to find the best hyperparameters.
        """
        param_distributions = self.get_param_distributions()
        if param_distributions is None:
            return None, None, None

        model = ModelFactory.get_model(self.model_name)
        if model is None:
            return None, None, None

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        random_search.fit(self.X_train, self.y_train)
        best_params = random_search.best_params_
        best_score = -random_search.best_score_
        logging.info(f"Random Search Best Params: {best_params}")
        logging.info(f"Random Search Best MSE: {best_score}")
        best_model = random_search.best_estimator_
        return best_model, best_params, best_score