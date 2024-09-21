import os
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    sentiment_analyzer = pipeline("sentiment-analysis", device=0)
except Exception as e:
    print(f"Error initializing pipeline with GPU: {e}")
    sentiment_analyzer = pipeline("sentiment-analysis", device=-1)


def evaluate_text(text):
    result = sentiment_analyzer(text)
    return result


text_to_evaluate = "The movie was fantastic with great visuals and storyline."
evaluation_result = evaluate_text(text_to_evaluate)
print(evaluation_result)
