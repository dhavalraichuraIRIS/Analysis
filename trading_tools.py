import yfinance as yf
import pandas as pd
import numpy as np
from crewai.tools import tool

@tool("Moving Average Crossover Backtester")
def backtest_ma_crossover(ticker: str, start_date: str, end_date: str, short_window: int = 50, long_window: int = 200) -> str:
    """
    Backtests a Moving Average Crossover strategy for a given stock ticker.
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        short_window: Moving average short window (default 50)
        long_window: Moving average long window (default 200)
    Returns: A statistical summary of the strategy's performance.
    """
    try:
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return f"Error: No data found for {ticker} between {start_date} and {end_date}."

        # Calculate Moving Averages
        data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
        data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

        # Generate Trading Signals (1 = Buy/Hold, 0 = Sell/Cash)
        data['Signal'] = 0.0
        data.loc[data.index[short_window:], 'Signal'] = np.where(
            data['SMA_Short'][short_window:] > data['SMA_Long'][short_window:], 1.0, 0.0
        )

        # Calculate Returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)

        # Drop NaNs
        data = data.dropna()

        # Compute Metrics
        cumulative_market_return = (1 + data['Daily_Return']).prod() - 1
        cumulative_strategy_return = (1 + data['Strategy_Return']).prod() - 1
        
        # Annualized Sharpe Ratio (assuming 252 trading days, risk-free rate = 0 for simplicity)
        daily_volatility = data['Strategy_Return'].std()
        sharpe_ratio = (data['Strategy_Return'].mean() / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0

        # Maximum Drawdown
        cumulative_returns = (1 + data['Strategy_Return']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        return (
            f"Backtest Results for {ticker} ({start_date} to {end_date}):\n"
            f"- Strategy Cumulative Return: {cumulative_strategy_return * 100:.2f}%\n"
            f"- Market Buy & Hold Return: {cumulative_market_return * 100:.2f}%\n"
            f"- Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"- Max Drawdown: {max_drawdown * 100:.2f}%\n"
            f"- Total Trading Days: {len(data)}"
        )
    except Exception as e:
        return f"Error during backtest execution: {str(e)}"