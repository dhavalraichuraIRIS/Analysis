from crewai import Agent

def quantitative_analyst_agent(llm, tools) -> Agent:
    return Agent(
        role="Quantitative Financial Analyst",
        goal="Run algorithmic trading backtests and analyze statistical performance.",
        backstory="You specialize in evaluating algorithmic trading strategies and benchmarking against buy-and-hold strategies.",
        verbose=True,
        tools=tools,
        llm=llm
    )

def market_sentiment_agent(llm, tools) -> Agent:
    return Agent(
        role="Market Sentiment Analyst",
        goal="Fetch recent news headlines and determine if the short-term media sentiment is Bullish, Bearish, or Neutral.",
        backstory="""You are a financial news analyst. You read recent headlines, identify potential 
                    catalysts (earnings, product launches, lawsuits), and gauge the overall market mood.""",
        verbose=True,
        tools=tools,
        llm=llm
    )

def strategy_evaluator_agent(llm) -> Agent:
    return Agent(
        role="Trading Strategy Evaluator",
        goal="Merge quantitative backtest metrics and qualitative news sentiment into a final investment report.",
        backstory="""You are the Head of Risk and Strategy. You combine hard math from the quants 
                    with soft sentiment data from the news desk to make final deployment decisions and generate clear, 
                    actionable markdown reports for portfolio managers.""",
        verbose=True,
        llm=llm
    )