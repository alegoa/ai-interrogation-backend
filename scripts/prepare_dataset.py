"""
scripts/prepare_dataset.py

Responsabilité unique :
- Lire les données brutes issues du moteur (data/raw/dataset.jsonl)
- Valider et filtrer les samples
- Séparer en train / validation
- Écrire data/processed/train.jsonl et val.jsonl

Aucune logique ML.
Aucune tokenisation.
"""

import json
import random
from pathlib import Path
from typing import Dict, List


RAW_PATH = Path("data/raw/dataset.jsonl")
PROCESSED_DIR = Path("data/processed")

TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "val.jsonl"

TRAIN_RATIO = 0.9
RANDOM_SEED = 42


# -----------------------
# Validation des samples
# -----------------------

REQUIRED_FIELDS = {
    "question": str,
    "alibi": str,
    "strategy": str,
    "fear": (int, float),
    "pressure": (int, float),
    "answer": str,
}


def is_valid_sample(sample: Dict) -> bool:
    # Champs requis
    for field, field_type in REQUIRED_FIELDS.items():
        if field not in sample:
            return False
        if not isinstance(sample[field], field_type):
            return False

    # Valeurs numériques cohérentes
    if not (0.0 <= sample["fear"] <= 1.0):
        return False
    if not (0.0 <= sample["pressure"] <= 1.0):
        return False

    # Texte non vide
    if len(sample["question"].strip()) < 5:
        return False
    if len(sample["answer"].strip()) < 5:
        return False

    return True


# -----------------------
# Chargement des données
# -----------------------

def load_raw_dataset(path: Path) -> List[Dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                if is_valid_sample(sample):
                    samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples


# -----------------------
# Main
# -----------------------

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {RAW_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_raw_dataset(RAW_PATH)

    if len(samples) == 0:
        raise RuntimeError("Aucun sample valide trouvé dans le dataset brut.")

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_RATIO)

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    with TRAIN_PATH.open("w", encoding="utf-8") as f_train:
        for s in train_samples:
            f_train.write(json.dumps(s, ensure_ascii=False) + "\n")

    with VAL_PATH.open("w", encoding="utf-8") as f_val:
        for s in val_samples:
            f_val.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("Dataset préparé avec succès")
    print(f"- Total samples : {len(samples)}")
    print(f"- Train : {len(train_samples)}")
    print(f"- Val   : {len(val_samples)}")
    print(f"Fichiers écrits dans : {PROCESSED_DIR}")


if __name__ == "__main__":
    main()