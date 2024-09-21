import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict
import weave


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

question = "What is the capital of France?"
candidate_response = "The capital of France is Berlin."
reference_answer = "The capital of France is Paris."


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


def get_combined_verdict(question, candidate_response, reference_answer):
    """
    Using majority vote for the final verdict
    """
    verdicts = []

    for model_name in MODELS:
        verdict = get_llm_verdict(
            question, candidate_response, reference_answer, model_name
        )
        verdicts.append(verdict)

    true_count = sum("True" in v for v in verdicts)
    false_count = sum("False" in v for v in verdicts)

    final_verdict = "True" if true_count > false_count else "False"
    return final_verdict, verdicts


def get_combined_verdict_with_explanations(
    question, candidate_response, reference_answer
):
    verdicts_and_explanations: Dict[str, str] = {}

    for model_name in MODELS:
        verdict = get_llm_verdict(
            question, candidate_response, reference_answer, model_name
        )
        verdicts_and_explanations.update({model_name: verdict})

    # Extract verdicts and count majority
    true_count = sum("True" in ve for ve in verdicts_and_explanations.values())
    false_count = sum("False" in ve for ve in verdicts_and_explanations.values())

    final_verdict = "True" if true_count > false_count else "False"
    return final_verdict, verdicts_and_explanations


# Example
final_verdict, verdicts_and_explanations = get_combined_verdict_with_explanations(
    question, candidate_response, reference_answer
)
for model, verdict in verdicts_and_explanations.items():
    print(f"Judge: {model}\nVerdict and Explanation: {verdict}\n")
print(f"Final Verdict: {final_verdict}")
