from ml.inference import generate_answer


def ask_llm(suspect,question) -> str:
    return generate_answer(
        suspect,question,
    )