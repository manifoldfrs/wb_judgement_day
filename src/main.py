import asyncio
import json
import os
from collections import defaultdict
from typing import Dict

import openai
import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score
from weave.flow.scorer import MultiTaskBinaryClassificationF1

from extract_data import load_hotpotqa

# Set environment variable to limit parallel workers
os.environ["WEAVE_PARALLELISM"] = "3"

load_dotenv()
weave.init("llm-judge-evaluation")


openai = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

hotpotqa_data = load_hotpotqa()

hotpotqa_user_annotations = [
    "True",
    "False",
    "False",
    "True",
    "False",
    "True",
    "False",
    "True",
    "False",
    "True",
    "False",
    "False",
    "False",
    "False",
    "True",
    "False",
    "True",
    "True",
    "False",
    "True",
    "False",
    "False",
    "False",
    "True",
    "True",
    "True",
    "True",
    "True",
    "True",
    "False",
]


# Define the prompt generation function
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


class LLMJudgeModel(weave.Model):
    name: str
    model_name: str

    @weave.op()
    async def predict(
        self, question: str, candidate_response: str, reference_answer: str
    ) -> Dict[str, str]:
        prompt = generate_prompt(question, candidate_response, reference_answer)
        response = await openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.001,
        )
        generated_text = response.choices[0].message.content.strip()
        # Extract verdict and explanation from the generated text
        if "True" in generated_text:
            verdict = "True"
        elif "False" in generated_text:
            verdict = "False"
        else:
            verdict = "Ambiguous"
        explanation = generated_text
        return {"verdict": verdict, "explanation": explanation}


class CandidateModel(weave.Model):
    name: str
    model_name: str

    @weave.op()
    async def predict(self, question: str) -> str:
        response = await openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": question}],
            max_tokens=150,
            temperature=0.001,
        )
        candidate_response = response.choices[0].message.content.strip()
        return candidate_response


candidate_model = CandidateModel(
    name="candidate-model",
    model_name="mistralai/mistral-large",
)

judge_models = [
    LLMJudgeModel(name="judge-mistral", model_name="mistralai/mistral-large"),
    LLMJudgeModel(name="judge-llama", model_name="meta-llama/llama-guard-2-8b"),
    LLMJudgeModel(name="judge-openai", model_name="openai/gpt-3.5-turbo-0125"),
]


@weave.op()
def judge_score(target: str, judge_output: Dict[str, str]) -> Dict[str, bool]:
    return {"correct": target == judge_output["verdict"]}


scorers = [MultiTaskBinaryClassificationF1(class_names=["True", "False"]), judge_score]


async def prepare_evaluation_examples():
    evaluation_examples = []
    results = []
    verdicts_dict = {model.name: [] for model in judge_models}
    verdicts_dict["user"] = []

    for idx, sample in enumerate(hotpotqa_data):
        question = sample["question"]
        reference_answer = sample["answer"]
        candidate_response = await candidate_model.predict(question)

        # Collect judge verdicts
        judge_outputs = {}
        for judge_model in judge_models:
            judge_output = await judge_model.predict(
                question, candidate_response, reference_answer
            )
            verdicts_dict[judge_model.name].append(judge_output["verdict"])
            judge_outputs[judge_model.name] = judge_output

        user_verdict = hotpotqa_user_annotations[idx]
        verdicts_dict["user"].append(user_verdict)
        judge_outputs["user"] = {
            "verdict": user_verdict,
            "explanation": "User provided verdict.",
        }

        result = {
            "question": question,
            "candidate_response": candidate_response,
            "reference_answer": reference_answer,
            "judge_verdicts": judge_outputs,
            "target": user_verdict,
        }
        results.append(result)
        evaluation_examples.append(result)

    return evaluation_examples, results, verdicts_dict


def compute_kappa_statistics(verdicts_dict):
    """
    Computes Cohen's Kappa statistics between each judge model and user annotations.

    Parameters:
    - verdicts_dict: Dictionary of verdict lists from each judge model and user.

    Returns:
    - A dictionary containing Cohen's Kappa values between each judge and the user.
    """
    user_verdicts = verdicts_dict["user"]
    kappa_results = {}

    for judge_name, judge_verdicts in verdicts_dict.items():
        if judge_name == "user":
            continue  # Skip comparison with self
        # Filter out ambiguous verdicts
        filtered_pairs = [
            (user_verdict, judge_verdict)
            for user_verdict, judge_verdict in zip(user_verdicts, judge_verdicts)
            if user_verdict != "Ambiguous" and judge_verdict != "Ambiguous"
        ]
        if not filtered_pairs:
            kappa = None
            print(f"No data to compute Cohen's Kappa between {judge_name} and user")
        else:
            user_filtered, judge_filtered = zip(*filtered_pairs)
            kappa = cohen_kappa_score(user_filtered, judge_filtered)
            print(f"Cohen's Kappa between {judge_name} and user: {kappa}")
        kappa_results[f"kappa_{judge_name}_user"] = kappa

    return kappa_results


async def main():
    evaluation_examples, results, verdicts_dict = await prepare_evaluation_examples()

    # Compute Kappa statistics between judges and user annotations
    kappa_results = compute_kappa_statistics(verdicts_dict)

    # Save results and kappa statistics to JSON files
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    with open("kappa_results.json", "w", encoding="utf-8") as f:
        json.dump(kappa_results, f, ensure_ascii=False, indent=4)

    print("Results and Kappa statistics have been saved to JSON files.")

    evaluation = weave.Evaluation(
        name="judge_evaluation",
        dataset=evaluation_examples,
        scorers=scorers,
    )

    for judge_model in judge_models:
        print(f"Evaluating {judge_model.name}")
        await evaluation.evaluate(judge_model)

    # Optionally, you can evaluate your own annotations if the framework supports it
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
