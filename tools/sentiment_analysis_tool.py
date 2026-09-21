import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from typing import List, Dict, Union
import json
from tqdm import tqdm
import tensorflow as tf
import numpy as np

class SentimentAnalyzer:
    # Initialize the sentiment analyzer with model and tokenizer
    def __init__(self, device: str = None):
        # Determine the device to use (GPU or CPU)
        self.device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
        print(f"Using device: {self.device}")

        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name, from_pt=False  # Load in TensorFlow format
        )
        # self.model.eval()  # Set model to evaluation mode

        self.labels = {0: "negative", 1: "neutral", 2: "positive"}

    # Preprocess the text by combining and tokenizing title and review
    def preprocess_text(self, title: str, review: str) -> dict:
        """Combine and tokenize title and review text."""
        # Combine title and review text
        combined_text = f"{title} - {review}"
        return self.tokenizer(
            combined_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="tf",
        )

    # Analyze the sentiment of a review using the NLP model
    def analyze_text(self, title: str, review: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment based purely on text content."""
        # Preprocess the text
        inputs = self.preprocess_text(title, review)

        # Run the model prediction
        outputs = self.model(inputs)

        # Convert logits to probabilities using softmax
        probabilities = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]

        # Get predicted sentiment label and confidence
        sentiment_label = self.labels[np.argmax(probabilities)]
        confidence = np.max(probabilities)

        return {"Label": sentiment_label, "Confidence": round(float(confidence), 3)}

        # # Get model prediction without gradient calculation
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     probabilities = torch.softmax(
        #         outputs.logits, dim=1
        #     )  # Calculate probabilities
        #     # print(torch.argmax(probabilities, dim=1))
        #     sentiment_label = self.labels[
        #         torch.argmax(probabilities, dim=1).item()
        #     ]  # Get the predicted sentiment label
        #     confidence = torch.max(probabilities).item()  # Get the confidence score

        # return {"Label": sentiment_label, "Confidence": round(confidence, 3)}

    # Analyze sentiment for a list of reviews
    def analyze_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """Analyze sentiment for a list of reviews."""
        # Initialize a list to store analyzed reviews
        analyzed_reviews = []

        for review in tqdm(eval(reviews), desc="Analyzing reviews"):
            # Analyze sentiment for each review
            sentiment = self.analyze_text(review["Title"], review["Review"])
            # Create a dictionary for the analyzed review
            analyzed_review = {**review, "Sentiment": sentiment}
            analyzed_reviews.append(analyzed_review)  # Add analyzed review to the list

        return analyzed_reviews

# Function to analyze sentiments from a JSON input
def analyse_sentiments(input_json):
    # Initialize the sentiment analyzer
    analyzer = SentimentAnalyzer()
    # Analyze reviews to get sentiment data
    analyzed_reviews = analyzer.analyze_reviews(input_json)

    # Count the number of positive, negative, and neutral sentiments
    sentiment_counts = {
        "positive": sum(
            1 for r in analyzed_reviews if r["Sentiment"]["Label"] == "positive"
        ),
        "negative": sum(
            1 for r in analyzed_reviews if r["Sentiment"]["Label"] == "negative"
        ),
        "neutral": sum(
            1 for r in analyzed_reviews if r["Sentiment"]["Label"] == "neutral"
        ),
    }

    return analyzed_reviews, sentiment_counts