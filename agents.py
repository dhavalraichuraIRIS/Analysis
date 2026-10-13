from crewai import Agent

def quantitative_analyst_agent(llm, tools) -> Agent:
    return Agent(
        role="Quantitative Financial Analyst",
        goal="Run algorithmic trading backtests and analyze statistical performance.",
        backstory="""You are a seasoned quantitative analyst at a top-tier hedge fund. 
                    You specialize in evaluating algorithmic trading strategies, 
                    analyzing volatility, and benchmarking against buy-and-hold strategies.""",
        verbose=True,
        tools=tools,
        llm=llm
    )

def strategy_evaluator_agent(llm) -> Agent:
    return Agent(
        role="Trading Strategy Evaluator",
        goal="Format backtest metrics into professional, executive-ready investment reports.",
        backstory="""You are the Head of Risk and Strategy. Your job is to take raw 
                    backtest statistics from the quant team and translate them into 
                    clear, actionable markdown reports for portfolio managers.""",
        verbose=True,
        llm=llm
    )