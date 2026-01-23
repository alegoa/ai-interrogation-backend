"""
ml/format.py

Responsabilité unique :
- Transformer un sample brut (issu de data/raw/dataset.jsonl)
  en (prompt, response) utilisable pour l'entraînement et l'inférence.

Aucune tokenisation.
Aucune logique ML.
Format figé v1.
"""

from typing import Dict


def format_prompt(sample: Dict) -> str:
    """
    Construit le prompt textuel à partir d'un sample brut.

    Attendu dans sample :
    - question: str
    - alibi: str
    - strategy: str  # truth | lie_vague | deflect
    - fear: float    # 0.0 à 1.0
    - pressure: float  # 0.0 à 1.0
    """

    prompt = f"""
Tu es un suspect interrogé par la police dans un jeu d'enquête.

Tu dois répondre en restant strictement dans ton rôle.

Contexte :
- Alibi : {sample['alibi']}
- Stratégie : {sample['strategy']}
- Niveau de peur (0.0 à 1.0) : {sample['fear']}
- Niveau de pression (0.0 à 1.0) : {sample['pressure']}

Règles de comportement :
- Ne jamais avouer un crime.
- Rester cohérent avec l'alibi.
- Adapter le ton et la formulation au niveau de peur et de pression.
- Plus la peur est élevée, plus la réponse peut être hésitante ou floue.
- Plus la pression est élevée, plus la réponse peut être courte ou défensive.
- Répondre en français.
- Réponse courte (1 à 3 phrases).
- Ne jamais mentionner explicitement la peur ou la pression.

Question :
{sample['question']}

Réponse :
""".strip()

    return prompt


def format_sample(sample: Dict) -> Dict:
    """
    Retourne un dict contenant :
    - prompt: str
    - response: str
    """
    return {
        "prompt": format_prompt(sample),
        "response": sample["answer"].strip()
    }