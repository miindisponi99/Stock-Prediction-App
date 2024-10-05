import streamlit as st
import numpy as np

import copy
import random
import logging
import multiprocessing

from deap import base, creator, tools, algorithms
from typing import Tuple, Optional
from sklearn.model_selection import cross_val_score
from models.model_factory import ModelFactory
from sklearn.gaussian_process.kernels import ConstantKernel, RBF


class GeneticAlgorithmOptimizer:
    def __init__(self, model_name: str, X, y):
        self.model_name = model_name
        self.X = X
        self.y = y
        self.toolbox = None
        self.best_individual = None
        self.best_params = {}
        self.best_score = None

    def evaluate_model(self, individual) -> Tuple[float]:
        """
        Evaluate the model with the given hyperparameters.
        """
        try:
            params = self.individual_to_params(individual)
            model = ModelFactory.get_model(self.model_name, params)
            if model is None:
                return (np.inf,)
            mse = -np.mean(
                cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
            )
            return (mse,)
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            return (np.inf,)

    def individual_to_params(self, individual):
        """
        Convert individual list to parameter dictionary based on the model.
        """
        params = {}
        if self.model_name == "Random Forest":
            params = {
                'n_estimators': int(individual[0]),
                'max_depth': int(individual[1]),
                'min_samples_split': int(individual[2]),
                'min_samples_leaf': int(individual[3])
            }
        elif self.model_name == "XGBoost":
            params = {
                'n_estimators': int(individual[0]),
                'max_depth': int(individual[1]),
                'learning_rate': float(individual[2]),
                'subsample': float(individual[3])
            }
        elif self.model_name == "LightGBM":
            params = {
                'n_estimators': int(individual[0]),
                'max_depth': int(individual[1]),
                'learning_rate': float(individual[2]),
                'num_leaves': int(individual[3])
            }
        elif self.model_name == "SVR":
            params = {
                'C': float(individual[0]),
                'epsilon': float(individual[1]),
                'kernel': individual[2]
            }
        elif self.model_name == "KNN":
            params = {
                'n_neighbors': int(individual[0]),
                'weights': individual[1]
            }
        elif self.model_name == "ElasticNet":
            params = {
                'alpha': float(individual[0]),
                'l1_ratio': float(individual[1])
            }
        elif self.model_name == "Decision Tree":
            params = {
                'max_depth': int(individual[0]),
                'min_samples_split': int(individual[1]),
                'min_samples_leaf': int(individual[2])
            }
        elif self.model_name == "Ridge":
            params = {
                'alpha': float(individual[0]),
                'solver': individual[1]
            }
        elif self.model_name == "Lasso":
            params = {
                'alpha': float(individual[0]),
                'selection': individual[1]
            }
        elif self.model_name == "SGDRegressor":
            params = {
                'loss': individual[0],
                'penalty': individual[1],
                'alpha': float(individual[2]),
                'learning_rate': individual[3]
            }
        elif self.model_name == "CatBoost":
            params = {
                'depth': int(individual[0]),
                'learning_rate': float(individual[1]),
                'iterations': int(individual[2]),
                'l2_leaf_reg': float(individual[3])
            }
        elif self.model_name == "Gaussian Process":
            params = {
                'kernel': None,
                'alpha': float(individual[1])
            }
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=float(individual[0]),
                length_scale_bounds=(1e-2, 1e2)
            )
            params['kernel'] = kernel
        elif self.model_name == "MLPRegressor":
            params = {
                'hidden_layer_sizes': individual[0],
                'activation': individual[1],
                'solver': individual[2],
                'alpha': float(individual[3]),
                'learning_rate': individual[4]
            }
        return params

    def setup_toolbox(self, num_attributes=10):
        """
        Setup the Genetic Algorithm toolbox.
        """
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        attr_names = ()
        if self.model_name == "Random Forest":
            self.toolbox.register("n_estimators", random.randint, 50, 300)
            self.toolbox.register("max_depth", random.randint, 5, 50)
            self.toolbox.register("min_samples_split", random.randint, 2, 10)
            self.toolbox.register("min_samples_leaf", random.randint, 1, 4)
            num_attributes = 4
            attr_names = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf")
        elif self.model_name == "XGBoost":
            self.toolbox.register("n_estimators", random.randint, 50, 300)
            self.toolbox.register("max_depth", random.randint, 3, 10)
            self.toolbox.register("learning_rate", random.uniform, 0.01, 0.2)
            self.toolbox.register("subsample", random.uniform, 0.6, 1.0)
            num_attributes = 4
            attr_names = ("n_estimators", "max_depth", "learning_rate", "subsample")
        elif self.model_name == "LightGBM":
            self.toolbox.register("n_estimators", random.randint, 50, 300)
            self.toolbox.register("max_depth", random.randint, 5, 50)
            self.toolbox.register("learning_rate", random.uniform, 0.01, 0.2)
            self.toolbox.register("num_leaves", random.randint, 20, 150)
            num_attributes = 4
            attr_names = ("n_estimators", "max_depth", "learning_rate", "num_leaves")
        elif self.model_name == "SVR":
            self.toolbox.register("C", random.uniform, 0.1, 100.0)
            self.toolbox.register("epsilon", random.uniform, 0.01, 1.0)
            self.toolbox.register("kernel", random.choice, ['linear', 'poly', 'rbf', 'sigmoid'])
            num_attributes = 3
            attr_names = ("C", "epsilon", "kernel")
        elif self.model_name == "KNN":
            self.toolbox.register("n_neighbors", random.randint, 1, 30)
            self.toolbox.register("weights", random.choice, ['uniform', 'distance'])
            num_attributes = 2
            attr_names = ("n_neighbors", "weights")
        elif self.model_name == "ElasticNet":
            self.toolbox.register("alpha", random.uniform, 0.01, 1.0)
            self.toolbox.register("l1_ratio", random.uniform, 0.0, 1.0)
            num_attributes = 2
            attr_names = ("alpha", "l1_ratio")
        elif self.model_name == "Decision Tree":
            self.toolbox.register("max_depth", random.randint, 5, 50)
            self.toolbox.register("min_samples_split", random.randint, 2, 10)
            self.toolbox.register("min_samples_leaf", random.randint, 1, 4)
            num_attributes = 3
            attr_names = ("max_depth", "min_samples_split", "min_samples_leaf")
        elif self.model_name == "Ridge":
            self.toolbox.register("alpha", random.uniform, 0.01, 10.0)
            self.toolbox.register("solver", random.choice, ['auto', 'svd', 'cholesky', 'lsqr', 'sag'])
            num_attributes = 2
            attr_names = ("alpha", "solver")
        elif self.model_name == "Lasso":
            self.toolbox.register("alpha", random.uniform, 0.0001, 1.0)
            self.toolbox.register("selection", random.choice, ['cyclic', 'random'])
            num_attributes = 2
            attr_names = ("alpha", "selection")
        elif self.model_name == "SGDRegressor":
            self.toolbox.register("loss", random.choice, ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
            self.toolbox.register("penalty", random.choice, ['l2', 'l1', 'elasticnet'])
            self.toolbox.register("alpha", random.uniform, 0.0001, 1.0)
            self.toolbox.register("learning_rate", random.choice, ['constant', 'optimal', 'invscaling', 'adaptive'])
            num_attributes = 4
            attr_names = ("loss", "penalty", "alpha", "learning_rate")
        elif self.model_name == "CatBoost":
            self.toolbox.register("depth", random.randint, 4, 10)
            self.toolbox.register("learning_rate", random.uniform, 0.01, 0.3)
            self.toolbox.register("iterations", random.randint, 100, 1000)
            self.toolbox.register("l2_leaf_reg", random.uniform, 1, 10)
            num_attributes = 4
            attr_names = ("depth", "learning_rate", "iterations", "l2_leaf_reg")
        elif self.model_name == "Gaussian Process":
            self.toolbox.register("length_scale", random.uniform, 0.1, 10.0)
            self.toolbox.register("alpha_gp", random.uniform, 1e-10, 1e-2)
            num_attributes = 2
            attr_names = ("length_scale", "alpha_gp")
        elif self.model_name == "MLPRegressor":
            self.toolbox.register("hidden_layer_sizes", random.choice, [(50,), (100,), (100, 50), (100, 100)])
            self.toolbox.register("activation", random.choice, ['relu', 'tanh', 'logistic'])
            self.toolbox.register("solver", random.choice, ['adam', 'lbfgs', 'sgd'])
            self.toolbox.register("alpha", random.uniform, 0.0001, 1.0)
            self.toolbox.register("learning_rate", random.choice, ['constant', 'invscaling', 'adaptive'])
            num_attributes = 5
            attr_names = ("hidden_layer_sizes", "activation", "solver", "alpha", "learning_rate")
        else:
            st.error("Unsupported model for GA optimization.")
            logging.error("Unsupported model for GA optimization.")
            return None

        if num_attributes:
            attr_functions = tuple(getattr(self.toolbox, attr) for attr in attr_names)
            self.toolbox.register("individual", tools.initCycle, creator.Individual,
                                    attr_functions,
                                    n=1)
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
            if self.model_name in ["Random Forest", "XGBoost", "LightGBM", "Decision Tree", "CatBoost"]:
                self.toolbox.register("mutate", tools.mutUniformInt, 
                                        low=[50, 5, 2, 1], 
                                        up=[300, 50, 10, 4], 
                                        indpb=0.2)
            elif self.model_name == "SVR":
                self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            elif self.model_name in ["KNN", "Lasso", "ElasticNet"]:
                self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
            elif self.model_name == "SGDRegressor":
                self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
            elif self.model_name == "Gaussian Process":
                self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
            elif self.model_name == "MLPRegressor":
                self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
            elif self.model_name in ["Ridge", "Lasso"]:
                self.toolbox.register("mutate", tools.mutUniformFloat, 
                                        low=[0.01, 0.0], 
                                        up=[10.0, 1.0], 
                                        indpb=0.2)
            else:
                self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

            self.toolbox.register("select", tools.selTournament, tournsize=3)
            logging.info("Configured Genetic Algorithm parameters for " + self.model_name)

    def setup_ga(self):
        """
        Initialize the GA toolbox.
        """
        self.setup_toolbox()

    def optimize(self, population_size=50, generations=30, cxpb=0.7, mutpb=0.2) -> Optional[Tuple[object, dict, float]]:
        """
        Run the Genetic Algorithm to optimize hyperparameters.
        """
        try:
            population = self.toolbox.population(n=population_size)
            logging.info(f"Initialized population with size {population_size}.")
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            hof = tools.HallOfFame(1)
        
            for gen in range(generations):
                offspring = algorithms.varAnd(population, self.toolbox, cxpb, mutpb)
                fits = map(self.evaluate_model, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                population = self.toolbox.select(offspring, len(offspring))
                record = stats.compile(population)
                logging.info(f"Generation {gen}: {record}")
                if hasattr(st, 'progress'):
                    st.progress(gen / generations)
        
            hof.update(population)
            self.best_individual = hof[0]
            self.best_params = self.individual_to_params(self.best_individual)
            self.best_score = self.best_individual.fitness.values[0]
            logging.info(f"Best individual: {self.best_individual}, Fitness: {self.best_score}")
            best_model = ModelFactory.get_model(self.model_name, self.best_params)
            if best_model is not None:
                best_model.fit(self.X, self.y)
                return best_model, self.best_params, self.best_score
            else:
                logging.error("Failed to instantiate the best model.")
                return None, None, None
        except Exception as e:
            logging.error(f"Error during GA optimization: {e}")
            st.error(f"Error during GA optimization: {e}")
            return None, None, None