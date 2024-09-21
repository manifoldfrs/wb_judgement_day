import os
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

login(token=os.getenv("HUGGINGFACE_TOKEN"))


MODELS = [
    "mistralai/Mistral-Small-Instruct-2409",
    "meta-llama/Llama-Guard-3-8B",
    "openai/gpt-4o",
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

    if model_name.startswith("openai/"):
        # Use OpenAI's GPT-4 via API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        generated_text = response["choices"][0]["message"]["content"].strip()
    else:
        # Use Hugging Face model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = model.generate(input_ids, max_new_tokens=150)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
