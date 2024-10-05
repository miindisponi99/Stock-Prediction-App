
import streamlit as st
import numpy as np
import lightgbm as lgb

import optuna
import logging

from typing import Optional, Tuple, Dict
from sklearn.model_selection import cross_val_score
from models.model_factory import ModelFactory
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class BayesianOptimizationOptimizer:
    def __init__(self, model_name: str, X: np.ndarray, y: np.ndarray):
        """
        Initialize the Bayesian Optimization optimizer.

        Parameters:
        - model_name: Name of the machine learning model.
        - X: Feature matrix.
        - y: Target vector.
        """
        self.model_name = model_name
        self.X = X
        self.y = y
        self.study = None
        self.best_params = None
        self.best_score = None

    def define_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Define the objective function for Optuna based on the selected model.

        Parameters:
        - trial: Optuna trial object.

        Returns:
        - mse: Mean Squared Error to minimize.
        """
        if self.model_name == "Random Forest":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 5, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == "XGBoost":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        elif self.model_name == "LightGBM":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 10, 30)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            num_leaves = trial.suggest_int('num_leaves', 31, 150)
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == "SVR":
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            model = SVR(
                C=C,
                epsilon=epsilon,
                kernel=kernel
            )
        elif self.model_name == "KNN":
            n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                n_jobs=-1
            )
        elif self.model_name == "ElasticNet":
            alpha = trial.suggest_float('alpha', 0.01, 1.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=42,
                max_iter=10000
            )
        elif self.model_name == "Decision Tree":
            max_depth = trial.suggest_int('max_depth', 5, 50)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        elif self.model_name == "Ridge":
            alpha = trial.suggest_float('alpha', 0.01, 10.0, log=True)
            solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sag'])
            model = Ridge(
                alpha=alpha,
                solver=solver,
                random_state=42
            )
        elif self.model_name == "Lasso":
            alpha = trial.suggest_float('alpha', 0.0001, 1.0, log=True)
            selection = trial.suggest_categorical('selection', ['cyclic', 'random'])
            model = Lasso(
                alpha=alpha,
                selection=selection,
                random_state=42,
                max_iter=10000
            )
        elif self.model_name == "SGDRegressor":
            loss = trial.suggest_categorical('loss', ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
            penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
            alpha = trial.suggest_float('alpha', 0.0001, 1.0, log=True)
            learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive'])
            model = SGDRegressor(
                loss=loss,
                penalty=penalty,
                alpha=alpha,
                learning_rate=learning_rate,
                random_state=42,
                max_iter=1000,
                tol=1e-3
            )
        elif self.model_name == "CatBoost":
            depth = trial.suggest_int('depth', 4, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            iterations = trial.suggest_int('iterations', 100, 1000)
            l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)
            model = CatBoostRegressor(
                depth=depth,
                learning_rate=learning_rate,
                iterations=iterations,
                l2_leaf_reg=l2_leaf_reg,
                random_state=42,
                verbose=0
            )
        elif self.model_name == "Gaussian Process":
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
            alpha_gp = trial.suggest_float('alpha_gp', 1e-10, 1e-2, log=True)
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha_gp,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_name == "MLPRegressor":
            hidden_layer_sizes = trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (100, 50), (100, 100)])
            activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
            solver = trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd'])
            alpha_mlp = trial.suggest_float('alpha', 0.0001, 1.0, log=True)
            learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha_mlp,
                learning_rate=learning_rate,
                random_state=42,
                max_iter=1000
            )
        else:
            st.error("Unsupported model selected for Bayesian Optimization.")
            return np.inf

        mse = -np.mean(cross_val_score(model, self.X, self.y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1))
        return mse

    def optimize_hyperparameters(self, n_trials: int = 30, progress_placeholder: Optional[st.delta_generator.DeltaGenerator] = None) -> Tuple[Optional[object], Optional[Dict], Optional[float]]:
        """
        Optimize hyperparameters using Bayesian Optimization with Optuna.

        Parameters:
        - n_trials: Number of Optuna trials.
        - progress_placeholder: Streamlit placeholder to display progress.

        Returns:
        - best_model: Trained model with best hyperparameters.
        - best_params: Best hyperparameters found.
        - best_score: Best MSE score achieved.
        """
        try:
            self.study = optuna.create_study(direction='minimize')

            def optuna_callback(study, trial):
                if progress_placeholder:
                    progress = trial.number / n_trials
                    progress_placeholder.progress(progress)
            
            self.study.optimize(self.define_objective, n_trials=n_trials, callbacks=[optuna_callback])
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            logging.info(f"Bayesian Optimization Best Params: {self.best_params}")
            logging.info(f"Bayesian Optimization Best MSE: {self.best_score}")
            best_model = ModelFactory.get_model(self.model_name, self.best_params)
            if best_model is None:
                return None, None, None
            best_model.fit(self.X, self.y)
            logging.info(f"Trained model: {best_model}")
    
            return best_model, self.best_params, self.best_score
        except Exception as e:
            st.error(f"Error during Bayesian Optimization: {e}")
            logging.error(f"Error during Bayesian Optimization: {e}")
            return None, None, None