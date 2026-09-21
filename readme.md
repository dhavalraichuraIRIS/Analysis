# Sentiment-analysis-crewai Project

## Project Overview

The sentiment-analysis-crewai project is designed to analyze customer reviews. It processes review data, maps columns, analyzes sentiments, and generates markdown reports. The project leverages machine learning models and various tools to provide insights into customer feedback.

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone [repository-url]
   cd sentiment-analysis-crewai
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project, execute the `main.py` script with the path to your review file:

```bash
python main.py
```

Ensure that the necessary environment variables, such as `GROQ_API_KEY`, are set before running the script.

## Dependencies

The project requires the following Python packages:

- `google_play_scraper==1.2.7`
- `pandas==2.2.3`
- `crewai==0.95.0`
- `black==24.10.0`
- `torch==2.5.1`
- `transformers==4.48.0`
- `tqdm==4.67.1`
- `boto3`
- `tensorflow`
- `langchain_groq`

## Environment Variables

- `GROQ_API_KEY`: API key for accessing the Groq language model.

## Acknowledgments

- Thanks to the contributors and the open-source community for their support.
