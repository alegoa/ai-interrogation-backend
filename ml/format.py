def format_input(example: dict) -> str:
    emotion = example["emotion"]

    return (
        "[SUSPECT]\n"
        f"name={example['suspect']}\n"
        f"strategy={example['strategy']}\n"
        f"fear={emotion['fear']:.2f}\n"
        f"pressure={emotion['pressure']:.2f}\n"
        f"confidence={emotion['confidence']:.2f}\n\n"
        "[QUESTION]\n"
        f"{example['question']}\n\n"
        "[ANSWER]\n"
    )