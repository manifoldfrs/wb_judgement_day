import asyncio
import json
import os
from collections import defaultdict
from typing import Dict, List

import openai
import statsmodels.stats.inter_rater as ir
import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.metrics import cohen_kappa_score
from weave.flow.scorer import MultiTaskBinaryClassificationF1

from extract_data import load_hotpotqa, load_triviaqa, load_truthfulqa

# Set environment variable to limit parallel workers
# os.environ["WEAVE_PARALLELISM"] = "3"

load_dotenv()
weave.init("llm-judge-evaluation")


openai = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")
)

hotpotqa_data = load_hotpotqa()


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


# Define the LLMJudgeModel class
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


# Define the CandidateModel class
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
    model_name="mistralai/mistral-7b-instruct:free",
)

judge_models = [
    LLMJudgeModel(
        name="judge-mistral", model_name="mistralai/mistral-7b-instruct:free"
    ),
    LLMJudgeModel(name="judge-llama", model_name="meta-llama/llama-3.1-70b-instruct"),
    LLMJudgeModel(name="judge-openai", model_name="openai/gpt-3.5-turbo-instruct"),
]


@weave.op()
def judge_score(target: str, judge_output: Dict[str, str]) -> Dict[str, bool]:
    return {"correct": target == judge_output["verdict"]}


scorers = [MultiTaskBinaryClassificationF1(class_names=["True", "False"]), judge_score]


async def prepare_evaluation_examples():
    evaluation_examples = []
    results = []
    verdicts_dict = {judge_model.name: [] for judge_model in judge_models}

    for sample in hotpotqa_data:
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

        result = {
            "question": question,
            "candidate_response": candidate_response,
            "reference_answer": reference_answer,
            "judge_verdicts": judge_outputs,
            "target": ("True" if reference_answer == candidate_response else "False"),
        }
        results.append(result)

        evaluation_examples.append(result)

    return evaluation_examples, results, verdicts_dict


def compute_kappa_statistics(verdicts_dict):
    """
    Computes Fleiss' Kappa and pairwise Cohen's Kappa statistics for the given verdicts.

    Parameters:
    - verdicts_dict: Dictionary of verdict lists from each judge model.

    Returns:
    - A dictionary containing Fleiss' Kappa and Cohen's Kappa values.
    """
    verdicts_list = list(verdicts_dict.values())
    num_judges = len(verdicts_list)
    num_items = len(verdicts_list[0])

    # Prepare data for Fleiss' Kappa
    fleiss_data = []
    for i in range(num_items):
        verdict_counts = defaultdict(int)
        for judge_verdicts in verdicts_list:
            verdict = judge_verdicts[i]
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
    print(f"Fleiss' Kappa: {fleiss_kappa}")

    # Compute pairwise Cohen's Kappa
    kappa_results = {"fleiss_kappa": fleiss_kappa}
    judge_names = list(verdicts_dict.keys())
    num_judges = len(judge_names)
    for i in range(num_judges):
        for j in range(i + 1, num_judges):
            v1 = verdicts_dict[judge_names[i]]
            v2 = verdicts_dict[judge_names[j]]
            # Filter out ambiguous verdicts
            filtered_v1_v2 = [
                (v1_k, v2_k)
                for v1_k, v2_k in zip(v1, v2)
                if v1_k != "Ambiguous" and v2_k != "Ambiguous"
            ]
            if not filtered_v1_v2:
                kappa = None
                print(
                    f"No data to compute Cohen's Kappa between {judge_names[i]} and {judge_names[j]}"
                )
            else:
                filtered_v1, filtered_v2 = zip(*filtered_v1_v2)
                kappa = cohen_kappa_score(filtered_v1, filtered_v2)
                print(
                    f"Cohen's Kappa between {judge_names[i]} and {judge_names[j]}: {kappa}"
                )
            key = f"kappa_{judge_names[i]}_{judge_names[j]}"
            kappa_results[key] = kappa

    return kappa_results


async def main():
    evaluation_examples, results, verdicts_dict = await prepare_evaluation_examples()

    # Since we don't have ground truth, we skip scoring and directly compute Kappa statistics
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

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
