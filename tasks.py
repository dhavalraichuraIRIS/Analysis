from crewai import Task, Agent

def create_backtest_task(agent: Agent, ticker: str, start_date: str, end_date: str) -> Task:
    return Task(
        description=f"""
            Use the 'Moving Average Crossover Backtester' tool to backtest 
            a strategy for {ticker} from {start_date} to {end_date}.
            
            Evaluate the returned statistics (Strategy Return, Market Return, 
            Sharpe Ratio, and Max Drawdown). Determine if the algorithmic strategy 
            outperformed the standard market buy-and-hold approach and assess its risk profile.
        """,
        expected_output="A summary of the backtest metrics and a brief risk-adjusted performance analysis.",
        agent=agent,
    )

def create_investment_report_task(agent: Agent) -> Task:
    return Task(
        description="""
            Review the backtest performance analysis provided by the Quantitative Analyst.
            Create a professional markdown report summarizing the findings.
            
            The report must include:
            1. Executive Summary
            2. Performance Metrics (Strategy Return vs Market Return, Sharpe Ratio, Max Drawdown)
            3. Risk vs. Reward Evaluation
            4. Final Recommendation (Deploy, Modify, or Reject the strategy)
        """,
        expected_output="A polished markdown investment report.",
        agent=agent,
        output_file="algorithmic_trading_report.md"
    )