import json
import random
from pathlib import Path
from typing import Dict
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
OUTPUT_PATH = Path("data/raw/dataset.jsonl")

N_SAMPLES = 5

STRATEGIES = ["truth", "lie_vague", "deflect"]

QUESTIONS = [
    "Où étiez-vous entre 21h et 22h ?",
    "Avez-vous vu la victime ce soir-là ?",
    "Quelqu’un peut-il confirmer où vous étiez ?",
    "Pourquoi étiez-vous près du lieu du crime ?",
    "Vous êtes-vous disputé avec la victime ?",
    "Que faisiez-vous ce soir-là ?",
    "Pourquoi votre téléphone apparaît-il dans cette zone ?"
]

ALIBIS = [
    "J’étais chez moi en train de regarder la télévision.",
    "Je conduisais sans destination précise.",
    "J’étais chez un ami.",
    "Je travaillais tard.",
    "Je suis sorti faire un tour.",
    "Je suis resté chez moi toute la soirée."
]

def build_prompt(sample: Dict) -> str:
    return f"""
Tu écris des dialogues pour un jeu d’enquête policière.

Le personnage est un suspect interrogé par la police.
Il doit répondre EN RESTANT DANS SON RÔLE, en respectant la stratégie indiquée.

Règles impératives :
- Ne jamais avouer un crime.
- Rester cohérent avec l’alibi donné.
- Adapter le ton et la formulation en fonction du niveau de peur et de pression.
- Plus la peur est élevée, plus la réponse peut être hésitante ou floue.
- Plus la pression est élevée, plus la réponse peut être courte, défensive ou sèche.
- La réponse doit faire 1 à 3 phrases maximum.
- Ne jamais mentionner explicitement la peur ou la pression.
- Ne pas utiliser un langage trop littéraire.

Contexte :
- Stratégie : {sample['strategy']}
- Niveau de peur (0.0 à 1.0) : {sample['fear']}
- Niveau de pression (0.0 à 1.0) : {sample['pressure']}
- Alibi : {sample['alibi']}

Question :
{sample['question']}

Réponse :
""".strip()

from openai import OpenAI
import time

client = OpenAI()


def call_llm(prompt: str) -> str:
    """
    Appel OpenAI pour générer une réponse courte et contrôlée.
    Optimisé pour génération de dataset (coût / qualité).
    """

    for attempt in range(3):  # retries simples
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_output_tokens=120,   # réponse courte
                temperature=0.8,         # variété contrôlée
                top_p=0.95,
            )

            text = response.output_text.strip()

            if not text:
                raise ValueError("Réponse vide")

            return text

        except Exception as e:
            print(f"[LLM ERROR] tentative {attempt + 1}: {e}")
            time.sleep(1)

    raise RuntimeError("Échec de génération après 3 tentatives")

# 5. Génération d’un échantillon
def generate_sample() -> Dict:
    strategy = random.choice(STRATEGIES)

    fear = round(random.uniform(0.0, 1.0), 2)
    pressure = round(random.uniform(0.0, 1.0), 2)

    question = random.choice(QUESTIONS)
    alibi = random.choice(ALIBIS)

    sample = {
        "question": question,
        "alibi": alibi,
        "strategy": strategy,
        "fear": fear,
        "pressure": pressure,
    }

    prompt = build_prompt(sample)
    answer = call_llm(prompt).strip()
    if len(answer) > 500:
        raise ValueError("Réponse trop longue")

    sample["answer"] = answer
    return sample

# 6. Boucle principale
def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(N_SAMPLES):
            try:
                sample = generate_sample()
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                if i % 50 == 0:
                    print(f"{i}/{N_SAMPLES} échantillons générés")

            except Exception as e:
                print(f"Erreur à l’échantillon {i} : {e}")

if __name__ == "__main__":
    main()
