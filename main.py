import os
import dotenv
import sys
import json
from crewai import Crew, LLM
# from langchain_groq import ChatGroq
from utils import FileProcessor, process_data_dict,get_rating_distribution
from agents import *
from tasks import *
from sentiment_analysis_tool import *

dotenv.load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def understand_reviews(file_path: str):
    # Initialize the language model
    llm = LLM(
        model="openai/openai/gpt-oss-20b",          # Double prefix!
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",  
        max_tokens=2500
    )

    # Create a file processor for reading the file
    file_processor = FileProcessor(file_path)  
    # Create a data mapping agent
    mapping_agent = data_format_agent(llm=llm)

    # Read the file into a DataFrame
    df = file_processor.read_file().head(1000) # Limit rows for performance

    # Create and execute the mapping task
    mapping_task = create_column_mapping_task(mapping_agent, df, llm=llm)
    crew = Crew(agents=[mapping_agent], tasks=[mapping_task], verbose=False)

    # Execute the crew to get the column mapping
    result = crew.kickoff()
    # Parse the column mapping result
    column_mapping = json.loads(result.raw)
    print("Detected column mapping:", json.dumps(column_mapping, indent=2))

    # Validate the mapping
# Validate the mapping
    required_fields = {"DateOfReview", "Rating", "Review", "Title"}
    
    for field in required_fields:
        if field not in column_mapping or column_mapping[field] is None:
            print(f"\n❌ ERROR: LLM failed to map the '{field}' column. Please check if your Excel file has this data.")
            print(f"LLM returned: {json.dumps(column_mapping, indent=2)}")
            raise ValueError(f"Missing required field mapping: {field}")

    # Process the data with the detected mapping
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
    llm = LLM(
        model="openai/openai/gpt-oss-20b",          # Double prefix!
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",  
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




# # test = understand_reviews(file_path="/Users/deepakjhanji/Documents/demand_model/voice-of-customer/apple-reviews.xlsx")
# if __name__ == "__main__":
#     # Get the file path from command line arguments
#     # file_path = "/Users/deepakjhanji/Documents/demand_model/voice-of-customer/apple-reviews.xlsx"
#     file_path = "/Users/deepakjhanji/Downloads/deriv_googleplay.xlsx"
#     # Process the reviews and get updated data, sentiments, and ratings
#     reviews_with_sentiment, sentiment_distribution, rating_distribution = understand_reviews(file_path)
#     # Generate and print markdown reports
#     print(generate_markdowns(reviews_with_sentiment, sentiment_distribution, rating_distribution))


import sys
from PyQt6.QtWidgets import QApplication, QFileDialog

def select_file():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv if sys.argv else [""])

    # Define filter options: combined supported types first, then individual, then all
    filter_string = (
        "Supported Files (*.xlsx *.xls *.csv);;"
        "Excel Files (*.xlsx *.xls);;"
        "CSV Files (*.csv);;"
        "All Files (*)"
    )

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Excel or CSV File",
        "",
        filter_string,
        options=QFileDialog.Option.DontUseNativeDialog,
    )

    return file_path or None

if __name__ == "__main__":
    file_path = select_file()

    # Exit cleanly if the user clicked "Cancel"
    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    # Process reviews
    reviews_with_sentiment, sentiment_distribution, rating_distribution = (
        understand_reviews(file_path)
    )
    print(
        generate_markdowns(
            reviews_with_sentiment, sentiment_distribution, rating_distribution
        )
    )