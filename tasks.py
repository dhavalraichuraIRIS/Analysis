from crewai import Task, Agent

def create_backtest_task(agent: Agent, ticker: str, start_date: str, end_date: str) -> Task:
    return Task(
        description=f"Backtest a strategy for {ticker} from {start_date} to {end_date} and summarize the returns and risk.",
        expected_output="A summary of the backtest metrics.",
        agent=agent,
    )

def create_sentiment_task(agent: Agent, ticker: str) -> Task:
    return Task(
        description=f"""
            Use the 'Recent News Fetcher' tool to get the last 5 days of news for {ticker}.
            Read the headlines and write a short paragraph determining if the current sentiment is 
            Bullish, Bearish, or Neutral. Highlight any major catalysts mentioned in the headlines.
        """,
        expected_output="A short sentiment analysis paragraph with an overall Bullish/Bearish/Neutral rating.",
        agent=agent,
    )

def create_investment_report_task(agent: Agent) -> Task:
    return Task(
        description="""
            Review the quantitative backtest metrics AND the short-term news sentiment analysis.
            Create a professional markdown report including:
            1. Executive Summary
            2. Quantitative Performance (Strategy vs Market, Sharpe, Drawdown)
            3. Qualitative Market Sentiment (Last 5 Days Outlook)
            4. Final Recommendation (Should we deploy this strategy given the current news cycle?)
        """,
        expected_output="A polished markdown investment report combining math and news.",
        agent=agent,
        output_file="algorithmic_trading_report.md"
    )