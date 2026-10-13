from crewai import Task, Agent
from typing import Dict, Optional, List
import pandas as pd


def create_column_mapping_task(agent: Agent, df: pd.DataFrame, llm) -> Task:
    sample_data = df.head(5).to_dict("records")
    columns = df.columns.tolist()

    return Task(
        description=f"""Analyze the following columns and their sample data to determine which columns 
                       should map to the required fields (DateOfReview, Rating, Review, Title).

                       Columns available: {columns}
                       Sample data: {sample_data}

                       Instructions:
                       - Map "DateOfReview" to the column representing the date.
                       - Map "Rating" to the column representing a numeric score or stars.
                       - Map "Review" to the main text/content of the review.
                       - Map "Title" to the review title or header.
                       - If a column clearly does not exist, use the closest match available.

                       Provide the mapping EXACTLY in this JSON format:
                       {{
                           "DateOfReview": "actual_column_name",
                           "Rating": "actual_column_name",
                           "Review": "actual_column_name",
                           "Title": "actual_column_name"
                       }}""",
        expected_output="A valid JSON object containing the 4 column mappings.",
        agent=agent,
        llm=llm
    )


# Function to create a data analysis task with specific parameters
def create_data_analysis_task(
        agent: Agent,
        rating_distribution: Dict,
        sentiment_distribution: Dict,
        reviews_list: List = [],
) -> Task:
    # Initialize a list to store text reviews
    text_reviews = []
    # Check if there are multiple reviews
    if len(reviews_list) > 1:
        # Extract review text from each review
        for review in reviews_list:
            text_reviews.append(review.get("Review", ""))

    # Return a Task object with a detailed description and expected output
    return Task(
        description=f"""
            Analyze the following review data:
            - Total Reviews: {rating_distribution['total_reviews']}
            - Rating Distribution: {rating_distribution['rating_counts']}
            - Sentiment Distribution: {sentiment_distribution}

            - All the reviews {text_reviews}

            Prepare a detailed breakdown including:
            1. Overall rating and sentiment insights
            2. Key trends and patterns
            3. Most common themes in reviews
            4. Recommendations based on the analysis
            5. Most frequent words 
            6. Most frequest phrases
            """,
        expected_output="""
            A comprehensive analysis report including:
            - Summary of key findings
            - Detailed breakdown of ratings and sentiments
            - Identified trends and patterns
            - Strategic recommendations
            """,
        agent=agent,
    )

# Function to create a reporting task for generating markdown reports
def create_reporting_task(agent: Agent) -> Task:
    # Return a Task object with a description and expected output for the report
    return Task(
        description="""
        Convert the data analysis into a professional markdown report:
        - Use clear headings and subheadings
        - Include charts/tables where appropriate
        - Highlight key insights
        - Provide a concise executive summary
        - Include actionable recommendations
        """,
        expected_output="""
            A well-structured markdown report with:
            - Clear executive summary
            - Detailed data visualization
            - Actionable insights
            - Formatted in professional markdown
            """,
        agent=agent,  # Assign the agent responsible for the task
        output_file="review_analysis_report_deepseek.md",  # Specify the output file for the report
    )