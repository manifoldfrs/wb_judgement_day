import os
from openai import OpenAI
from huggingface_hub import login
from dotenv import load_dotenv
import weave


load_dotenv()

weave.init("together-weave")

openai = OpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

login(token=os.getenv("HUGGINGFACE_TOKEN"))


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
        temperature=0.1,
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


# Get combined verdict from multiple models
final_verdict, individual_verdicts = get_combined_verdict(
    question, candidate_response, reference_answer
)
print(f"Final Verdict: {final_verdict}")
for idx, verdict in enumerate(individual_verdicts, 1):
    print(f"Model {idx} Verdict: {verdict}")


def get_combined_verdict_with_explanations(
    question, candidate_response, reference_answer
):
    verdicts_and_explanations = []

    for model_name in MODELS:
        verdict = get_llm_verdict(
            question, candidate_response, reference_answer, model_name
        )
        verdicts_and_explanations.append(verdict)

    # Extract verdicts and count majority
    true_count = sum("True" in ve for ve in verdicts_and_explanations)
    false_count = sum("False" in ve for ve in verdicts_and_explanations)

    final_verdict = "True" if true_count > false_count else "False"
    return final_verdict, verdicts_and_explanations


# Example
final_verdict, verdicts_and_explanations = get_combined_verdict_with_explanations(
    question, candidate_response, reference_answer
)
print(f"Final Verdict: {final_verdict}")
for idx, ve in enumerate(verdicts_and_explanations, 1):
    print(f"Judge {idx} Verdict and Explanation: {ve}")
