import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.jsonl")

def log_interaction(
    suspect: str,
    question: str,
    strategy: str,
    state: dict,
    answer: str
):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "suspect": suspect,
        "question": question,
        "strategy": strategy,
        "emotion": {
            "fear": state["fear"],
           
            "pressure": state["pressure"]
        },
        "answer": answer
    }

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")