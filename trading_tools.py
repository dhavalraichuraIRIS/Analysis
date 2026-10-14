import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai.tools import tool

@tool("Moving Average Crossover Backtester")
def backtest_ma_crossover(ticker: str, start_date: str, end_date: str, short_window: int = 50, long_window: int = 200) -> str:
    """Backtests a Moving Average Crossover strategy for a given stock ticker."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return f"Error: No data found for {ticker}."

        data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
        data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
        data['Signal'] = 0.0
        data.loc[data.index[short_window:], 'Signal'] = np.where(
            data['SMA_Short'][short_window:] > data['SMA_Long'][short_window:], 1.0, 0.0
        )
        data['Daily_Return'] = data['Close'].pct_change()
        data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)
        data = data.dropna()

        cumulative_market = (1 + data['Daily_Return']).prod() - 1
        cumulative_strategy = (1 + data['Strategy_Return']).prod() - 1
        
        daily_volatility = data['Strategy_Return'].std()
        sharpe_ratio = (data['Strategy_Return'].mean() / daily_volatility) * np.sqrt(252) if daily_volatility > 0 else 0

        cum_returns = (1 + data['Strategy_Return']).cumprod()
        max_drawdown = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()

        return (
            f"Backtest Results for {ticker} ({start_date} to {end_date}):\n"
            f"- Strategy Cumulative Return: {cumulative_strategy * 100:.2f}%\n"
            f"- Market Buy & Hold Return: {cumulative_market * 100:.2f}%\n"
            f"- Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"- Max Drawdown: {max_drawdown * 100:.2f}%\n"
        )
    except Exception as e:
        return f"Error during backtest execution: {str(e)}"

@tool("Recent News Fetcher")
def fetch_recent_news(ticker: str) -> str:
    """
    Fetches news headlines for a stock ticker from the last 5 days.
    Args: ticker: Stock symbol (e.g., 'AAPL')
    Returns: A list of recent news headlines and publication dates.
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        if not news_items:
            return f"No recent news found for {ticker}."

        five_days_ago = datetime.now() - timedelta(days=5)
        recent_news = []
        
        for item in news_items:
            # yfinance returns timestamps in seconds
            pub_date = datetime.fromtimestamp(item['providerPublishTime'])
            if pub_date >= five_days_ago:
                recent_news.append(f"- {pub_date.strftime('%Y-%m-%d')}: {item['title']} (Source: {item.get('publisher', 'Unknown')})")

        if not recent_news:
            return f"No news found for {ticker} in the last 5 days."
            
        return f"Recent News for {ticker}:\n" + "\n".join(recent_news)
    except Exception as e:
        return f"Error fetching news: {str(e)}"