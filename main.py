import sys
import json
from crewai import Crew, LLM
from crewai.flow.flow import Flow, listen, start
from tools.utils import FileProcessor, process_data_dict,get_rating_distribution
from tools.sentiment_analysis_tool import *
# from agentlib.data_formatting_agent import create_analysis_agent
# from tasklib.data_formatting_tasks import create_column_mapping_task
# from tools.sentiment_analysis_tool import analyse_sentiments
from agentlib.agents import *
from tasklib.tasks import *
import boto3
import os
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
load_dotenv()

REGION_NAME = 'us-east-1'
SECRET_NAME = "postgres/ml-models/"

def get_openai_api_key(secret_name, specific_key):
    client = boto3.client('secretsmanager', region_name='us-east-1')
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_string = response["SecretString"]

        # Parse the secret JSON and extract the specific key
        secret_dict = json.loads(secret_string)
        return secret_dict.get(specific_key, None)
    except Exception as e:
        print(f"Error fetching secret: {e}")
        return None


# openai_api_key = get_openai_api_key("lead-genie/api-keys", "OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = openai_api_key

api_key = os.getenv("GROQ_API_KEY")

def understand_reviews(file_path: str):
    # Initialize components
    # Initialize the language model
    # llm = LLM(model="gpt-4o", api_key=openai_api_key)
    # llm = ChatGroq(model_name="llama-3.3-70b-specdec", api_key=api_key)
    llm = LLM(
        api_key=api_key,
        # model="groq/llama-3.3-70b-specdec",
        model="groq/deepseek-r1-distill-llama-70b",
        max_tokens=500
    )
    # Create a file processor for reading the file
    file_processor = FileProcessor(file_path)
    # Create a data mapping agent
    mapping_agent = data_format_agent(llm=llm)

    # Read the file
    # Read the file into a DataFrame
    df = file_processor.read_file()

    # Create and execute the mapping task
    # Create a task for column mapping
    mapping_task = create_column_mapping_task(mapping_agent, df, llm=llm)
    # Initialize the crew with agents and tasks
    crew = Crew(agents=[mapping_agent], tasks=[mapping_task], verbose=False)

    # Get the column mapping from the agent
    # Execute the crew to get the column mapping
    result = crew.kickoff()
    # Parse the column mapping result
    column_mapping = json.loads(result.raw)
    print("Detected column mapping:", json.dumps(column_mapping, indent=2))

   # Validate the mapping
    # Define required fields for validation
    required_fields = {"DateOfReview", "Rating", "Review", "Title"}
    if not all(field in column_mapping for field in required_fields):
        raise ValueError("Missing required fields in the column mapping")

    # Process the data with the detected mapping
    # Process the data using the detected column mapping - cleaning and pre-processing of text
    processed_data = process_data_dict(df, column_mapping)
    # Convert processed data to JSON format
    json_output = json.dumps(processed_data, indent=2)
    # Analyze sentiments in the processed data
    print(json_output)

    reviews_with_sentiment, sentiment_distribution = analyse_sentiments(json_output)
    # Return the processed data with sentiment and rating distribution
    return (
        reviews_with_sentiment,
        sentiment_distribution,
        get_rating_distribution(reviews_with_sentiment),
    )

# Function to generate markdown reports from review data
def generate_markdowns(reviews_with_sentiment, sentiment_distribution, rating_distribution):
    # Initialize the language model
    # llm = LLM(model="gpt-4o")
    llm = LLM(
        api_key=api_key,
        # model="groq/llama-3.3-70b-specdec",
        model="groq/deepseek-r1-distill-llama-70b",
        max_tokens=500
    )

    # Initialize the crew with agents and tasks for reporting
    crew = Crew(
        agents=[data_analysis_agent(llm=llm), technical_writer_agent(llm=llm)],
        tasks=[
            create_data_analysis_task(data_analysis_agent(llm=llm),rating_distribution,sentiment_distribution,reviews_with_sentiment),
            create_reporting_task(technical_writer_agent(llm=llm)),
        ],
        verbose=False,
    )

    # Execute the crew
    # Execute the crew to generate reports
    result = crew.kickoff()
    return result




# test = understand_reviews(file_path="/Users/deepakjhanji/Documents/demand_model/voice-of-customer/apple-reviews.xlsx")
if __name__ == "__main__":
    # Get the file path from command line arguments
    # file_path = "/Users/deepakjhanji/Documents/demand_model/voice-of-customer/apple-reviews.xlsx"
    file_path = "/Users/deepakjhanji/Downloads/deriv_googleplay.xlsx"
    # Process the reviews and get updated data, sentiments, and ratings
    reviews_with_sentiment, sentiment_distribution, rating_distribution = understand_reviews(file_path)
    # Generate and print markdown reports
    print(generate_markdowns(reviews_with_sentiment, sentiment_distribution, rating_distribution))