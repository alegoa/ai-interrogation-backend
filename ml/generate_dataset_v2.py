import json
import random
import time
from pathlib import Path
from typing import Dict

from openai import OpenAI

# =========================
# CONFIG
# =========================

OUTPUT_PATH = Path("data/dataset_v2newnew.jsonl")
N_SAMPLES = 1500
SLEEP_BETWEEN_CALLS = 0.5  # pour éviter le rate limit

MODEL_NAME = "gpt-4o-mini"  # ou autre modèle OpenAI

client = OpenAI()

# =========================
# POOLS DE DONNÉES (CANON ENTRAÎNEMENT)
# =========================

JOBS = [
    "chauffeur",
    "infirmier",
    "étudiant",
    "domestique",
    "technicien",
    "employé administratif",
    "commerçant",
]

PERSONALITIES = [
    "discret et réservé",
    "méfiant et peu bavard",
    "nerveux sous pression",
    "calme en apparence",
    "Poli, posé, coopératif",
    "S’emporte vite",
    "Séduisant, drôle",
    "Dépressif, fatigué",
    "Logique, méthodique"
]

RESIDENCES = [
    "centre-ville",
    "banlieue",
    "quartier nord",
    "quartier résidentiel",
]

BACKGROUND_FACTS = [
    "Vit seul",
    "Vit en couple",
    "A une routine quotidienne stable",
    "Horaires de travail irréguliers",
    "Relations sociales limitées",
    "Connaissait la victime de vue",
    "Fréquente peu ses voisins",
]

ALIBIS = [
    "J’étais chez moi toute la soirée.",
    "Je travaillais tard.",
    "Je suis sorti faire un tour.",
    "J’étais chez un ami.",
    "Je regardais la télévision chez moi.",
]

QUESTIONS = [
    "Où étiez-vous ce soir-là ?",
    "Quel est votre métier ?",
    "Avec qui vivez-vous ?",
    "Connaissiez-vous la victime ?",
    "Que faisiez-vous entre 21h et 22h ?",
    "Pourquoi semblez-vous nerveux ?",
    "Pouvez-vous décrire votre routine quotidienne ?",
]

STRATEGIES = ["truth", "lie_vague", "deflect"]

# =========================
# PROMPT CANON (À NE PLUS CHANGER)
# =========================

def build_prompt(sample: Dict) -> str:
    identity = sample["identity"]
    background = "\n".join(f"- {b}" for b in sample["background"])

    return f"""
Tu écris des dialogues pour un jeu d’enquête policière réaliste.

Tu incarnes un suspect interrogé par la police.
Tu dois répondre STRICTEMENT dans le rôle du personnage décrit.

=====================
RÈGLES IMPÉRATIVES
=====================
- Ne jamais avouer un crime.
- Ne jamais inventer d’informations non fournies.
- Ne jamais inventer d’événements, de relations ou de preuves.
- Ne jamais changer d’identité, d’âge, de métier ou de personnalité.
- Ne jamais expliquer ton raisonnement.
- Ne jamais utiliser un langage littéraire ou théâtral.

=====================
COMPORTEMENT
=====================
- Tu réponds comme une personne réelle interrogée par la police.
- Tu adaptes le ton selon la stratégie, la peur et la pression.
- Plus la peur est élevée → hésitations, nervosité.
- Plus la pression est élevée → réponses plus courtes, défensives.
- Réponse entre 1 et 4 phrases maximum.
- Ne jamais mentionner explicitement la peur, la pression ou la stratégie.

=====================
INFORMATIONS DU PERSONNAGE
=====================
Âge : {identity["age"]}
Métier : {identity["job"]}
Personnalité : {identity["personality"]}
Lieu de résidence : {identity["residence"]}

Contexte personnel :
{background}

Alibi :
{sample["alibi"]}

=====================
ÉTAT ACTUEL
=====================
Stratégie : {sample["strategy"]}
Niveau de peur (0.0 à 1.0) : {sample["fear"]}
Niveau de pression (0.0 à 1.0) : {sample["pressure"]}

=====================
QUESTION
=====================
{sample["question"]}

=====================
RÉPONSE
=====================
""".strip()

# =========================
# APPEL OPENAI
# =========================

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Tu es un suspect dans un jeu d’enquête policière."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# =========================
# GÉNÉRATION D’UN SAMPLE
# =========================

def generate_sample() -> Dict:
    identity = {
        "age": random.randint(20, 65),
        "job": random.choice(JOBS),
        "personality": random.choice(PERSONALITIES),
        "residence": random.choice(RESIDENCES),
    }

    background = random.sample(
        BACKGROUND_FACTS,
        k=random.randint(3, 6)
    )

    sample = {
        "question": random.choice(QUESTIONS),
        "strategy": random.choice(STRATEGIES),
        "fear": round(random.uniform(0.0, 1.0), 2),
        "pressure": round(random.uniform(0.0, 1.0), 2),
        "identity": identity,
        "background": background,
        "alibi": random.choice(ALIBIS),
    }

    prompt = build_prompt(sample)
    answer = call_llm(prompt)

    sample["answer"] = answer
    return sample

# =========================
# MAIN
# =========================

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(N_SAMPLES):
            try:
                sample = generate_sample()
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                if i % 50 == 0:
                    print(f"{i}/{N_SAMPLES} samples générés")

                time.sleep(SLEEP_BETWEEN_CALLS)

            except Exception as e:
                print(f"[ERREUR] Sample {i}: {e}")

if __name__ == "__main__":
    main()
