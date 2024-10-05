# Stock-Prediction-App

## Project Overview

The **Stock Price Prediction App** is a comprehensive web application designed to forecast future stock prices using various machine learning models. Leveraging advanced hyperparameter optimization techniques, including Genetic Algorithms, Bayesian Optimization, Grid Search, and Random Search, the app ensures optimal model performance. Additionally, it incorporates backtesting functionalities to evaluate the effectiveness of prediction strategies in real-world trading scenarios.

**Key Features:**
- **Data Fetching & Preprocessing:** Retrieves historical stock data and processes it for analysis.
- **Feature Engineering:** Generates insightful features to enhance model predictions.
- **Model Optimization:** Utilizes multiple optimization techniques to fine-tune model hyperparameters.
- **Model Evaluation:** Assesses model performance using metrics like Mean Squared Error (MSE), R², and more.
- **Backtesting:** Simulates trading strategies based on model predictions to evaluate profitability.
- **Interactive Web Interface:** User-friendly interface built with Streamlit for seamless interaction.

## Repository Structure

- **app.py:** The main Streamlit application script that ties all components together and serves the web interface.
  
- **data/**: Contains modules related to data handling.
  - **data_fetcher.py:** Handles data retrieval from financial APIs or data sources.
  - **data_preprocessor.py:** Manages data cleaning, normalization, and preprocessing tasks.
  - **feature_engineer.py:** Generates and selects features for model training.
  
- **models/**: Encompasses model-related modules.
  - **model_factory.py:** A factory class to instantiate different machine learning models based on user selection.
  - **optimizers/**: Houses various hyperparameter optimization techniques.
    - **genetic_algorithm.py:** Implements Genetic Algorithm-based optimization.
    - **grid_search.py:** Implements Grid Search optimization.
    - **random_search.py:** Implements Random Search optimization.
    - **bayesian_optimization.py:** Implements Bayesian Optimization.
  
- **evaluation/**: Contains modules for model evaluation and backtesting.
  - **model_evaluator.py:** Evaluates model performance using various metrics.
  - **backtester.py:** Simulates trading strategies to assess profitability based on model predictions.
  
- **utils/**: Utility modules for logging and helper functions.
  - **logger.py:** Sets up logging configurations.
  - **helpers.py:** Contains helper functions for plotting and other miscellaneous tasks.
  
- **requirements.txt:** Lists all Python dependencies required to run the application.
  
- **LICENSE:** Specifies the licensing information for the project.
  
- **README.md:** This documentation file.

## How to Use

### Prerequisites

- **Python 3.8 or higher** installed on your machine.
- **Git** installed for cloning the repository.

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/miindisponi99/Stock-Prediction-App.git
   cd Stock-Prediction-App
   ```

### How to Use the Web Application

The **Stock Price Prediction App** offers an intuitive interface that allows users to effortlessly forecast stock prices and evaluate trading strategies. Follow the steps below to get started:

1. **Launch the Application:**
   - After setting up the environment and installing dependencies, run the app using:
     ```bash
     streamlit run app.py
     ```
   - This will open the application in your default web browser. If it doesn't open automatically, navigate to `http://localhost:8501`.

2. **Provide User Inputs:**
   - **Ticker Symbol:** Enter the stock ticker symbol (e.g., `AAPL` for Apple Inc.) in the designated input field.
   - **Date Range:** Select the start and end dates to define the historical data period for analysis.
   - **Model Selection:** Choose a machine learning model from the dropdown menu (e.g., Random Forest, XGBoost, SVR).
   - **Optimization Method:** Select a hyperparameter optimization technique (e.g., Genetic Algorithm, Grid Search).

3. **Run Prediction:**
   - Click the **"Run Prediction"** button in the sidebar to initiate the data processing, model training, and optimization workflow.
   - The application will display progress indicators during various stages like data fetching, preprocessing, feature engineering, optimization, evaluation, and backtesting.

4. **View Results:**
   - **Data Overview:** Review the latest entries of the processed dataset displayed in the main panel.
   - **Model Configuration:** Confirm the selected model and optimization method.
   - **Optimization Outcomes:** After optimization, view the best hyperparameters and the corresponding Mean Squared Error (MSE).
   - **Model Evaluation:** Examine performance metrics such as R², MSE, MAE, and visualize actual vs. predicted stock prices through interactive plots.
   - **Backtesting Results:** Assess the effectiveness of the prediction strategy compared to a Buy & Hold approach, including final capital and performance metrics.
   - **Download Options:** Utilize the download buttons to export performance metrics and trade details as CSV files for further analysis.

### Live Demo

Experience the app firsthand by visiting the live deployment:
[Stock Price Prediction App](https://stockpredictionmodels.streamlit.app)

## Interpreting the Outputs

Understanding the results generated by the **Stock Price Prediction App** is crucial for making informed trading decisions. Here's a brief guide on how to interpret the key outputs:

1. **Best Parameters and MSE:**
   - **Best Parameters:** These are the hyperparameters that yielded the optimal performance for the selected model during the optimization process.
   - **Mean Squared Error (MSE):** Indicates the average squared difference between the predicted and actual stock prices. A lower MSE signifies better model accuracy.

2. **Model Performance Metrics:**
   - **R² Score:** Represents the proportion of variance in the dependent variable predictable from the independent variables. Values closer to 1 indicate a better fit.
   - **Mean Absolute Error (MAE):** Measures the average magnitude of errors in a set of predictions, without considering their direction.
   - **Visualizations:** The Actual vs. Predicted plot helps in visually assessing how closely the model's predictions align with real stock prices.

3. **Backtesting Results:**
   - **Final Capital:** Shows the amount of capital after executing the trading strategy based on model predictions compared to the initial capital.
   - **Buy & Hold Comparison:** Provides a benchmark by comparing the model-based strategy's performance against a traditional Buy & Hold approach.
   - **Performance Metrics:** Includes metrics like total return, maximum drawdown, and Sharpe ratio to evaluate the risk and profitability of the trading strategy.
   - **Trade Details:** Lists individual trades executed during backtesting, including buy/sell actions, prices, and returns.

4. **Downloadable Reports:**
   - **Performance Metrics CSV:** Offers a downloadable file containing all the performance metrics for offline analysis or record-keeping.
   - **Trade Details CSV:** Provides a detailed log of all trades made during backtesting, useful for in-depth strategy evaluation.

**Tips for Interpretation:**
- **Compare Metrics:** Use the provided metrics to compare different models and optimization methods to identify the most effective combination.
- **Risk Assessment:** Analyze backtesting metrics to understand the risk associated with each trading strategy.
- **Continuous Improvement:** Use the insights gained from model evaluations and backtesting to refine your models and strategies for better future performance.

By following this guide, users can effectively utilize the **Stock Price Prediction App** to make data-driven trading decisions and optimize their investment strategies.

## Requirements
To install the required Python libraries, run the following command:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the Apache License 2.0


---

This README provides an overview of the Stock-Prediction-App repository, including its features, requirements, usage, and detailed descriptions of model deployment using streamlit.
