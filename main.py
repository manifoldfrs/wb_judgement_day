import json
import os
from collections import defaultdict
from typing import Dict, List

import statsmodels.stats.inter_rater as ir
import weave
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

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
    Provided Answer: {candidate_response}
    Reference Answer: {reference_answer}

    You are a helpful assistant acting as an impartial judge. You will be given a Question, a Reference
    Answer, and a Provided Answer. Your task is to judge whether the Provided Answer is correct by
    comparing it to the Reference Answer. If the Provided Answer is correct, choose 'True', otherwise
    choose 'False'.

    An answer counts as true if it describes the literal truth about the real world. True factual claims are
    usually supported by reliable, publicly available evidence. Answers that are only true according to
    a belief system or tradition (e.g., "Libras have friendly personalities") are counted as false. Provide
    a brief explanation for your decision.
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
    """
    LLMs-as-Judges Majority
    """
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

# Lists to store results and verdicts
results: List[Dict] = []
verdicts_model1 = []
verdicts_model2 = []
verdicts_model3 = []


# Helper function to filter ambiguous verdicts
def filter_ambiguous(verdicts1, verdicts2):
    return zip(
        *[
            (v1, v2)
            for v1, v2 in zip(verdicts1, verdicts2)
            if v1 != "Ambiguous" and v2 != "Ambiguous"
        ]
    )


# Process each question-answer pair
for pair in hotpotqa_data:
    question = pair["question"]
    reference_answer = pair["answer"]

    # Generate candidate response
    candidate_response = get_candidate_response(question)

    # Get verdicts from models
    final_verdict, verdicts_and_explanations = get_combined_verdict_with_explanations(
        question, candidate_response, reference_answer
    )

    # Collect verdicts for each model
    verdicts = {}
    for model_name, verdict in verdicts_and_explanations.items():
        if "True" in verdict:
            verdicts[model_name] = "True"
        elif "False" in verdict:
            verdicts[model_name] = "False"
        else:
            verdicts[model_name] = "Ambiguous"  # Handle ambiguous verdicts

    verdicts_model1.append(verdicts[MODELS[0]])
    verdicts_model2.append(verdicts[MODELS[1]])
    verdicts_model3.append(verdicts[MODELS[2]])

    # Prepare result
    result = {
        "question": question,
        "candidate_response": candidate_response,
        "reference_answer": reference_answer,
        "verdicts": verdicts_and_explanations,
        "final_verdict": final_verdict,
    }

    results.append(result)


def compute_kappa_statistics(verdicts_model1, verdicts_model2, verdicts_model3):
    """
    Computes Fleiss' Kappa and pairwise Cohen's Kappa statistics for the given verdicts.

    Parameters:
    - verdicts_model1, verdicts_model2, verdicts_model3: Lists of verdicts from each model.

    Returns:
    - A dictionary containing Fleiss' Kappa and Cohen's Kappa values.
    """
    # Prepare data for Fleiss' Kappa
    fleiss_data = []
    for v1, v2, v3 in zip(verdicts_model1, verdicts_model2, verdicts_model3):
        verdict_counts = defaultdict(int)
        for verdict in [v1, v2, v3]:
            verdict_counts[verdict] += 1
        # Include counts for all categories
        fleiss_data.append(
            [
                verdict_counts["False"],
                verdict_counts["True"],
                verdict_counts["Ambiguous"],
            ]
        )

    # Compute Fleiss' Kappa
    fleiss_kappa = ir.fleiss_kappa(fleiss_data, method="fleiss")
    print("Fleiss' Kappa:", fleiss_kappa)

    # Helper function to filter ambiguous verdicts
    def filter_ambiguous(verdicts1, verdicts2):
        return zip(
            *[
                (v1, v2)
                for v1, v2 in zip(verdicts1, verdicts2)
                if v1 != "Ambiguous" and v2 != "Ambiguous"
            ]
        )

    # Compute Cohen's Kappa between Model 1 and Model 2
    v1_12, v2_12 = filter_ambiguous(verdicts_model1, verdicts_model2)
    kappa_12 = cohen_kappa_score(list(v1_12), list(v2_12))
    print("Cohen's Kappa between Model 1 and Model 2:", kappa_12)

    # Compute Cohen's Kappa between Model 1 and Model 3
    v1_13, v3_13 = filter_ambiguous(verdicts_model1, verdicts_model3)
    kappa_13 = cohen_kappa_score(list(v1_13), list(v3_13))
    print("Cohen's Kappa between Model 1 and Model 3:", kappa_13)

    # Compute Cohen's Kappa between Model 2 and Model 3
    v2_23, v3_23 = filter_ambiguous(verdicts_model2, verdicts_model3)
    kappa_23 = cohen_kappa_score(list(v2_23), list(v3_23))
    print("Cohen's Kappa between Model 2 and Model 3:", kappa_23)

    return {
        "fleiss_kappa": fleiss_kappa,
        "kappa_12": kappa_12,
        "kappa_13": kappa_13,
        "kappa_23": kappa_23,
    }


# Compute Kappa statistics
kappa_results = compute_kappa_statistics(
    verdicts_model1, verdicts_model2, verdicts_model3
)

# Optional: Use kappa_results as needed
# For example, you can write them to a file or further analyze them
print("Computed Kappa Statistics:", kappa_results)

# Write results to a JSON file
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# Write Kappa results to a separate JSON file
with open("kappa_results.json", "w", encoding="utf-8") as f:
    json.dump(kappa_results, f, ensure_ascii=False, indent=4)
