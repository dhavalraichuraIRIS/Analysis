import os
import dotenv
from crewai import Crew, LLM

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_DISABLE_TRACING"] = "true"

from trading_tools import backtest_ma_crossover, fetch_recent_news
from agents import quantitative_analyst_agent, market_sentiment_agent, strategy_evaluator_agent
from tasks import create_backtest_task, create_sentiment_task, create_investment_report_task

dotenv.load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def run_trading_crew(ticker: str, start_date: str, end_date: str):
    llm = LLM(
        model="openai/openai/gpt-oss-20b", # Reverted to your working model
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",  
        max_tokens=3000
    )

    # Initialize Agents with their respective tools
    quant_agent = quantitative_analyst_agent(llm=llm, tools=[backtest_ma_crossover])
    sentiment_agent = market_sentiment_agent(llm=llm, tools=[fetch_recent_news])
    evaluator_agent = strategy_evaluator_agent(llm=llm)

    # Create Tasks
    backtest_task = create_backtest_task(quant_agent, ticker, start_date, end_date)
    sentiment_task = create_sentiment_task(sentiment_agent, ticker)
    report_task = create_investment_report_task(evaluator_agent)

    # Note: CrewAI runs sequentially by default, so it will:
    # 1. Run the backtest
    # 2. Read the news
    # 3. Write the combined report
    crew = Crew(
        agents=[quant_agent, sentiment_agent, evaluator_agent],
        tasks=[backtest_task, sentiment_task, report_task],
        verbose=True
    )

    print(f"Starting analysis for {ticker}...")
    return crew.kickoff()

if __name__ == "__main__":
    # Test with Apple
    final_report = run_trading_crew("AAPL", "2026-01-01", "2026-10-10")
    print("\n--- FINAL REPORT ---")
    print(final_report)