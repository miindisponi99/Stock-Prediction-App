import streamlit as st
import pandas as pd

import time

from data.data_fetcher import DataFetcher
from data.data_preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from models.model_factory import ModelFactory
from evaluation.model_evaluator import ModelEvaluator
from evaluation.backtester import Backtester
from utils.logger import setup_logger
from utils.helpers import plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    setup_logger()
    st.title("📈 Stock Price Prediction App")
    st.write("""### Predict future stock prices using various machine learning models and optimization techniques.""")
    st.sidebar.header("User Inputs")
    ticker = st.sidebar.text_input("Ticker Symbol (e.g., AAPL)", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    model_option = st.sidebar.selectbox(
        "Select Model",
        (
            "Random Forest", "XGBoost", "LightGBM", "SVR", "KNN", 
            "ElasticNet", "Decision Tree", "Ridge", "Lasso", 
            "SGDRegressor", "CatBoost", "Gaussian Process", 
            "MLPRegressor"
        )
    )
    optimization_option = st.sidebar.selectbox(
        "Select Optimization Method",
        ("Genetic Algorithm", "Grid Search", "Random Search", "Bayesian Optimization")
    )

    if st.sidebar.button("Run Prediction"):
        initial_capital = 100000
        with st.spinner('Fetching and processing data...'):
            fetcher = DataFetcher(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            data = fetcher.fetch_data()
            if data.empty:
                st.error("Data fetching failed. Please check the inputs.")
                return
            preprocessor = DataPreprocessor(data)
            data = preprocessor.preprocess()
            if data.empty:
                st.error("Data preprocessing failed.")
                return
            engineer = FeatureEngineer(data)
            data = engineer.engineer_features()
            if data.empty:
                st.error("Feature engineering failed.")
                return
            st.success("Data fetched and preprocessed successfully!")

        st.subheader("Data Overview")
        st.write(data.tail())
        feature_cols = [
            'Close_Lag1', 'Close_Lag2', 'MA_10', 'MA_50', 'RSI', 
            'MACD', 'MACD_Signal', 'MACD_Diff', 'Bollinger_High', 
            'Bollinger_Low', 'Bollinger_Width', 'Daily_Return', 
            'Day_of_Week', 'Month'
        ]
        X = data[feature_cols]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            shuffle=False
        )
        st.subheader("Model and Optimization Configuration")
        st.write(f"**Model Selected:** {model_option}")
        st.write(f"**Optimization Method Selected:** {optimization_option}")
        progress_placeholder = st.empty()
        final_model = None
        best_params = None
        best_score = None
        scaling_needed = model_option in ["ElasticNet", "Ridge", "Lasso", "SGDRegressor", "MLPRegressor", "Gaussian Process"]
        if scaling_needed:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.write("Applied StandardScaler to features.")
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # Hyperparameter Optimization and Model Training
        with st.spinner('Optimizing hyperparameters...'):
            start_time = time.time()
            if optimization_option == "Genetic Algorithm":
                from models.optimizers.genetic_algorithm import GeneticAlgorithmOptimizer
                optimizer = GeneticAlgorithmOptimizer(model_option, X_train_scaled, y_train)
                optimizer.setup_ga()
                best_model, best_params, best_score = optimizer.optimize(
                    population_size=50, 
                    generations=30, 
                    cxpb=0.7, 
                    mutpb=0.2
                )
                final_model = best_model
            elif optimization_option == "Grid Search":
                from models.optimizers.grid_search import GridSearchOptimizer
                optimizer = GridSearchOptimizer(model_option, X_train, y_train)
                best_model, best_params, best_score = optimizer.optimize()
                final_model = best_model
            elif optimization_option == "Random Search":
                from models.optimizers.random_search import RandomSearchOptimizer
                optimizer = RandomSearchOptimizer(model_option, X_train, y_train)
                best_model, best_params, best_score = optimizer.optimize(n_iter=20)
                final_model = best_model
            elif optimization_option == "Bayesian Optimization":
                from models.optimizers.bayesian_optimization import BayesianOptimizationOptimizer
                optimizer = BayesianOptimizationOptimizer(model_option, X_train.values, y_train.values)
                best_model, best_params, best_score = optimizer.optimize_hyperparameters(n_trials=30, progress_placeholder=progress_placeholder)
                final_model = best_model
            else:
                st.error("Unsupported optimization method selected.")
                return
            end_time = time.time()
            elapsed_time = end_time - start_time

        if final_model:
            st.success("Hyperparameter optimization and model training completed!")
            st.write(f"**Elapsed Time:** {elapsed_time:.2f} seconds")
            st.write(f"**Best Parameters:** {best_params}")
            if best_score is not None:
                st.write(f"**Best MSE:** {best_score:.4f}")
        else:
            st.error("Model training failed.")
            return

        # Model Evaluation
        with st.spinner('Evaluating model performance...'):
            evaluator = ModelEvaluator(final_model, X_test, y_test)
            model_metrics, predictions = evaluator.evaluate()
            if model_metrics:
                st.subheader("Model Performance Metrics")
                metrics_df = pd.DataFrame(list(model_metrics.items()), columns=['Metric', 'Value'])
                st.table(metrics_df)
            else:
                st.error("Failed to evaluate model performance.")

        # Plot Predictions
        st.subheader("Actual vs. Predicted Stock Prices")
        plot_predictions(
            dates=data['Date'][-len(y_test):],
            actual=y_test,
            predicted=predictions
        )

        # Backtesting
        with st.spinner('Running backtest...'):
            backtester = Backtester(
                dates=data['Date'][-len(y_test):],
                actual_prices=y_test,
                predicted_prices=predictions,
                initial_capital=initial_capital
            )
            final_capital, portfolio_values, bh_portfolio_values, backtest_metrics, buy_hold_metrics, trades_df = backtester.backtest_strategy()
            if final_capital:
                st.subheader("Backtesting Results")
                st.write(f"**Initial Capital:** ${initial_capital:,.2f}")
                st.write(f"**Final Capital after Backtesting (Model Strategy):** ${final_capital:,.2f}")
                st.write(f"**Final Capital after Backtesting (Buy & Hold):** ${bh_portfolio_values[-1]:,.2f}")

                st.subheader("Backtesting Performance Metrics")
                combined_metrics = {**backtest_metrics, **buy_hold_metrics}
                combined_metrics_df = pd.DataFrame(list(combined_metrics.items()), columns=['Metric', 'Value'])
                combined_metrics_df = combined_metrics_df.set_index('Metric')
                st.table(combined_metrics_df)

                with st.expander("View Trade Details"):
                    st.write(trades_df)

                # Download Performance Metrics
                metrics_csv = combined_metrics.copy()
                metrics_csv = pd.DataFrame(list(metrics_csv.items()), columns=['Metric', 'Value'])
                csv = metrics_csv.to_csv(index=False)
                st.download_button(
                    label="Download Performance Metrics as CSV",
                    data=csv,
                    file_name='performance_metrics.csv',
                    mime='text/csv',
                )

                # Download Trade Details
                if not trades_df.empty:
                    trades_csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="Download Trade Details as CSV",
                        data=trades_csv,
                        file_name='trade_details.csv',
                        mime='text/csv',
                    )
            else:
                st.error("Backtesting failed.")


if __name__ == "__main__":
    main()