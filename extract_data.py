import json
import re


def load_hotpotqa():
    with open("data/hotpotqa_30.json", "r", encoding="utf-8") as f:
        data = f.read()
    # Remove line numbers at the start of each line
    data = re.sub(r"^\d+\|", "", data, flags=re.MULTILINE)
    # Parse JSON data
    records = json.loads(data)
    # Extract question and answer from each record
    qa_pairs = []
    for record in records:
        question = record.get("question")
        answer = record.get("answer")
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs


def load_truthfulqa():
    qa_pairs = []
    with open("data/truthfulqa_30.json", "r", encoding="utf-8") as f:
        for line in f:
            # Remove line number
            line = re.sub(r"^\d+\|", "", line)
            record = json.loads(line)
            question = record.get("Question")
            answer = record.get("Best Answer")
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs


def load_triviaqa():
    qa_pairs = []
    with open("data/triviaqa_30.json", "r", encoding="utf-8") as f:
        for line in f:
            # Remove line number
            line = re.sub(r"^\d+\|", "", line)
            record = json.loads(line)
            question = record.get("question")
            answer = record.get("value")
            qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs
