import os
import dotenv
from crewai import Crew, LLM

# Force disable CrewAI tracing to prevent the 20-second delay
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACING"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

from trading_tools import backtest_ma_crossover
from agents import quantitative_analyst_agent, strategy_evaluator_agent
from tasks import create_backtest_task, create_investment_report_task

dotenv.load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def run_trading_crew(ticker: str, start_date: str, end_date: str):
    # Initialize the LLM using the double-prefix fallback for stability on Groq
    llm = LLM(
        model="openai/openai/mixtral-8x7b-32768",          
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",  
        max_tokens=3000
    )

    # Initialize tool list
    tools = [backtest_ma_crossover]

    # Initialize Agents
    quant_agent = quantitative_analyst_agent(llm=llm, tools=tools)
    evaluator_agent = strategy_evaluator_agent(llm=llm)

    # Create Tasks
    backtest_task = create_backtest_task(quant_agent, ticker, start_date, end_date)
    report_task = create_investment_report_task(evaluator_agent)

    # Assemble and run the Crew
    crew = Crew(
        agents=[quant_agent, evaluator_agent],
        tasks=[backtest_task, report_task],
        verbose=True
    )

    print(f"Starting backtest for {ticker}...")
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    # You will need to install yfinance if you haven't already:
    # uv pip install yfinance
    
    ticker_symbol = "AAPL"
    start = "2020-01-01"
    end = "2024-01-01"
    
    final_report = run_trading_crew(ticker_symbol, start, end)
    print("\n--- FINAL REPORT ---")
    print(final_report)