import os
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

question = "What is the capital of France?"
candidate_response = "The capital of France is Berlin."
reference_answer = "The capital of France is Paris."


def generate_prompt(question, candidate_response, reference_answer):
    return f"""
    You are an impartial judge. Here is the information:
    Question: {question}
    Candidate Response: {candidate_response}
    Reference Answer: {reference_answer}

    Your task is to evaluate whether the candidate's response is correct.
    If the answer is correct, respond with 'True'. If it is incorrect, respond with 'False'.
    Also, provide a brief explanation for your decision.
    """


def get_llm_verdict(question, candidate_response, reference_answer):
    prompt = generate_prompt(question, candidate_response, reference_answer)

    response = openai.chat.completions.create(
        model="gpt-4o",  # or any other available model
        messages=[
            {"role": "system", "content": "You are an impartial judge."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
    )

    return response.choices[0].message.content.strip()


# Example Usage
verdict = get_llm_verdict(question, candidate_response, reference_answer)
print(f"LLM Judge Verdict: {verdict}")


def get_combined_verdict(question, candidate_response, reference_answer):
    # Assume you have multiple instances or variations of the LLM
    verdicts = []

    for _ in range(3):  # Simulating 3 judges
        verdict = get_llm_verdict(question, candidate_response, reference_answer)
        verdicts.append(verdict)

    # Majority vote for the final verdict
    true_count = verdicts.count("True")
    false_count = verdicts.count("False")

    final_verdict = "True" if true_count > false_count else "False"
    return final_verdict, verdicts


# Get combined verdict from multiple judges
final_verdict, individual_verdicts = get_combined_verdict(
    question, candidate_response, reference_answer
)
print(f"Final Verdict: {final_verdict}")
print(f"Individual Verdicts: {individual_verdicts}")


def get_combined_verdict_with_explanations(
    question, candidate_response, reference_answer
):
    verdicts_and_explanations = []

    for _ in range(3):  # Simulating 3 judges
        prompt = generate_prompt(question, candidate_response, reference_answer)
        response = openai.chat.completions.create(
            model="gpt-4o",  # or any other available model
            messages=[
                {"role": "system", "content": "You are an impartial judge."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
        )
        verdict_and_explanation = response.choices[0].message.content.strip()
        verdicts_and_explanations.append(verdict_and_explanation)

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
