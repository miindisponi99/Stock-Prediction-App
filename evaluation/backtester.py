import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import logging

from scipy.stats import norm


class Backtester:
    def __init__(self, dates: pd.Series, actual_prices: pd.Series, predicted_prices: np.ndarray, initial_capital=100000):
        self.dates = dates
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.initial_capital = initial_capital

    def backtest_strategy(self):
        """
        Backtest a trading strategy based on model predictions and compare it with Buy and Hold.
        """
        try:
            if isinstance(self.actual_prices, np.ndarray):
                actual_prices = pd.Series(self.actual_prices, index=self.dates)
            else:
                actual_prices = self.actual_prices.copy()

            capital = self.initial_capital
            position = 0
            buy_price = 0
            shares = 0
            portfolio_values = []
            bh_capital = self.initial_capital
            bh_shares = bh_capital // actual_prices.iloc[0]
            bh_capital -= bh_shares * actual_prices.iloc[0]
            bh_portfolio_values = []
            trades = []
            trade_profits = []

            for i in range(len(self.predicted_prices)):
                current_price = actual_prices.iloc[i]
                predicted_price = self.predicted_prices[i]
                date = self.dates.iloc[i]

                # Buy signal
                if predicted_price > current_price and position == 0:
                    shares = capital // current_price
                    if shares > 0:
                        buy_price = current_price
                        capital -= shares * current_price
                        position = 1
                        trades.append({'Date': date, 'Type': 'Buy', 'Price': buy_price, 'Shares': shares})
                        logging.info(f"Buy on {date.date()} at price {buy_price:.2f} shares: {shares}")

                # Sell signal
                elif predicted_price <= current_price and position == 1:
                    sell_price = current_price
                    capital += shares * sell_price
                    profit = (sell_price - buy_price) * shares
                    trade_profits.append(profit)
                    trades.append({'Date': date, 'Type': 'Sell', 'Price': sell_price, 'Shares': shares, 'Profit': profit})
                    logging.info(f"Sell on {date.date()} at price {sell_price:.2f} shares: {shares}, Profit: {profit:.2f}")
                    shares = 0
                    position = 0

                if position == 1:
                    portfolio_value = capital + shares * current_price
                else:
                    portfolio_value = capital
                portfolio_values.append(portfolio_value)

                # Buy and Hold Portfolio
                bh_portfolio_value = bh_capital + bh_shares * current_price
                bh_portfolio_values.append(bh_portfolio_value)

            # Final Sell if holding position
            if position == 1:
                sell_price = actual_prices.iloc[-1]
                capital += shares * sell_price
                profit = (sell_price - buy_price) * shares
                trade_profits.append(profit)
                trades.append({'Date': self.dates.iloc[-1], 'Type': 'Sell', 'Price': sell_price, 'Shares': shares, 'Profit': profit})
                logging.info(f"Final Sell on {self.dates.iloc[-1].date()} at price {sell_price:.2f} shares: {shares}, Profit: {profit:.2f}")
                portfolio_values[-1] = capital

            # Final Buy and Hold Capital
            bh_capital += bh_shares * actual_prices.iloc[-1]
            bh_portfolio_values[-1] = bh_capital

            trades_df = pd.DataFrame(trades)
            model_returns = pd.Series(portfolio_values).pct_change().fillna(0)
            total_return = (capital - self.initial_capital) / self.initial_capital
            num_years = (self.dates.iloc[-1] - self.dates.iloc[0]).days / 365.25
            ann_r = (1 + total_return) ** (1 / num_years) - 1
            ann_vol = (model_returns.std()) * np.sqrt(252)
            negative_returns = model_returns[model_returns < 0]
            semidev = negative_returns.std() * np.sqrt(252) if not negative_returns.empty else 0.0
            skew = model_returns.skew()
            kurt = model_returns.kurtosis()
            hist_var5 = model_returns.quantile(0.05)
            z = norm.ppf(0.05)
            cf_var5 = (z * semidev + skew / 6 * (z**2 - 1) * semidev +
                       (kurt - 3) / 24 * (z**3 - 3*z) * semidev)
            hist_cvar5 = model_returns[model_returns <= hist_var5].mean() if not model_returns[model_returns <= hist_var5].empty else 0.0
            rovar5 = ann_r / abs(hist_var5) if hist_var5 != 0 else 0.0
            ann_sr = ann_r / ann_vol if ann_vol != 0 else 0.0
            ann_sortr = ann_r / semidev if semidev != 0 else 0.0
            max_drawdown = ((pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)) /
                           pd.Series(portfolio_values).cummax()).max()
            ann_cr = ann_r / abs(max_drawdown) if max_drawdown != 0 else 0.0
            np_wdd_ratio = (capital - self.initial_capital) / (self.initial_capital * max_drawdown) if max_drawdown != 0 else 0.0
            positive_returns = model_returns[model_returns > 0]
            tail_ratio = positive_returns.mean() / abs(hist_var5) if hist_var5 != 0 else 0.0
            wins = model_returns[model_returns > 0]
            losses = model_returns[model_returns <= 0]
            win_ratio = len(wins) / (len(wins) + len(losses)) if (len(wins) + len(losses)) > 0 else 0.0
            avg_gain = wins.mean() if not wins.empty else 0.0
            avg_loss = losses.mean() if not losses.empty else 0.0
            profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 0.0
            num_wins = len(wins)
            num_losses = len(losses)

            backtest_metrics = {
                "Annualized Return": round(ann_r, 4),
                "Annualized Volatility": round(ann_vol, 4),
                "Semi-Deviation": round(semidev, 4),
                "Skewness": round(skew, 4),
                "Kurtosis": round(kurt, 4),
                "Historic VaR (5%)": round(hist_var5, 4),
                "Cornish-Fisher VaR (5%)": round(cf_var5, 4),
                "Historic CVaR (5%)": round(hist_cvar5, 4),
                "Return on VaR": round(rovar5, 4),
                "Sharpe Ratio": round(ann_sr, 4),
                "Sortino Ratio": round(ann_sortr, 4),
                "Calmar Ratio": round(ann_cr, 4),
                "Net Profit to Worst Drawdown": round(np_wdd_ratio, 4),
                "Tail Ratio": round(tail_ratio, 4),
                "Win Ratio": round(win_ratio, 4),
                "Average Gain": round(avg_gain, 4),
                "Average Loss": round(avg_loss, 4),
                "Profit Factor": round(profit_factor, 4),
                "Number of Winning Trades": int(num_wins),
                "Number of Losing Trades": int(num_losses)
            }

            # Buy and Hold Metrics
            bh_returns = actual_prices.pct_change().fillna(0)
            bh_total_return = (bh_capital - self.initial_capital) / self.initial_capital
            bh_ann_r = (1 + bh_total_return) ** (1 / num_years) - 1
            bh_ann_vol = bh_returns.std() * np.sqrt(252)
            bh_ann_sr = bh_ann_r / bh_ann_vol if bh_ann_vol != 0 else 0.0
            buy_hold_metrics = {
                "Annualized Return (Buy & Hold)": round(bh_ann_r, 4),
                "Annualized Volatility (Buy & Hold)": round(bh_ann_vol, 4),
                "Sharpe Ratio (Buy & Hold)": round(bh_ann_sr, 4)
            }

            # Plot Portfolio Value Over Time
            plt.figure(figsize=(14,7))
            plt.plot(self.dates, portfolio_values, label='Model Strategy')
            plt.plot(self.dates, bh_portfolio_values, label='Buy and Hold', alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.title('Backtesting Portfolio Performance')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            logging.info("Plotted Backtesting Portfolio Performance.")

            # Plot Drawdown Over Time
            drawdown_negative = (pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)) / pd.Series(portfolio_values).cummax()
            drawdown_negative = -drawdown_negative
            plt.figure(figsize=(14,7))
            plt.plot(self.dates, drawdown_negative, label='Drawdown', color='red')
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.title('Drawdown Over Time')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            logging.info("Plotted Drawdown Over Time.")

            # Plot VaR Over Time
            window = 252
            model_returns = pd.Series(portfolio_values).pct_change().fillna(0)
            rolling_var5 = model_returns.rolling(window=window).quantile(0.05).fillna(method='bfill')

            plt.figure(figsize=(14,7))
            plt.plot(self.dates, rolling_var5, label='Rolling VaR (5%)', color='red')
            plt.xlabel('Date')
            plt.ylabel('VaR')
            plt.title('Value at Risk (5%) Over Time')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            logging.info("Plotted VaR Over Time.")

            return (capital, portfolio_values, bh_portfolio_values, backtest_metrics, buy_hold_metrics, trades_df)
        except Exception as e:
            logging.error(f"Error during backtesting: {e}")
            return None, None, None, None, None, None