import json
import os
from typing import Dict, List

import weave
from dotenv import load_dotenv
from openai import OpenAI

from extract_data import load_hotpotqa, load_triviaqa, load_truthfulqa

load_dotenv()
weave.init("together-weave")

openai = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/chatgpt-4o-latest",
]


def generate_prompt(question, candidate_response, reference_answer):
    return f"""
    Question: {question}
    Candidate Response: {candidate_response}
    Reference Answer: {reference_answer}

    Evaluate whether the candidate's response is correct.
    If correct, respond with 'True'.
    If incorrect, respond with 'False'.
    Provide a brief explanation.
    """


@weave.op()
def get_candidate_response(question, model_name="mistralai/mistral-7b-instruct:free"):
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": question}],
        max_tokens=150,
        temperature=0.001,
    )
    candidate_response = response.choices[0].message.content.strip()
    return candidate_response


@weave.op()
def get_llm_verdict(question, candidate_response, reference_answer, model_name):
    prompt = generate_prompt(question, candidate_response, reference_answer)
    response = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.001,
    )
    generated_text = response.choices[0].message.content.strip()
    return generated_text


def get_combined_verdict_with_explanations(
    question, candidate_response, reference_answer
):
    verdicts_and_explanations: Dict[str, str] = {}
    for model_name in MODELS:
        verdict = get_llm_verdict(
            question, candidate_response, reference_answer, model_name
        )
        verdicts_and_explanations[model_name] = verdict

    true_count = sum("True" in ve for ve in verdicts_and_explanations.values())
    false_count = sum("False" in ve for ve in verdicts_and_explanations.values())

    final_verdict = "True" if true_count > false_count else "False"
    return final_verdict, verdicts_and_explanations


# Load data from datasets
hotpotqa_data = load_hotpotqa()
triviaqa_data = load_triviaqa()
truthfulqa_data = load_truthfulqa()

# List to store results
results: List[Dict] = []

# Process each question-answer pair
# Use one dataset for now
for pair in hotpotqa_data:
    question = pair["question"]
    reference_answer = pair["answer"]

    # Generate candidate response using the specified model
    candidate_response = get_candidate_response(question)

    # Get verdicts from other models
    final_verdict, verdicts_and_explanations = get_combined_verdict_with_explanations(
        question, candidate_response, reference_answer
    )

    # Prepare result dictionary
    result = {
        "question": question,
        "candidate_response": candidate_response,
        "reference_answer": reference_answer,
        "verdicts": verdicts_and_explanations,
        "final_verdict": final_verdict,
    }

    # Append to results list
    results.append(result)

# Write results to a JSON file
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
