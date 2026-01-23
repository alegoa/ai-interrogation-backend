"""
ml/dataset.py

Responsabilité unique :
- Charger les données préparées (JSONL)
- Construire les inputs du modèle (input_ids, attention_mask, labels)
- Masquer les labels du prompt (-100)
- Fournir un torch.utils.data.Dataset propre

Aucune logique métier.
Aucune logique d'entraînement.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ml.format import format_sample


class InterrogationDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        """
        Args:
            data_path: chemin vers un fichier JSONL (train.jsonl / val.jsonl)
            tokenizer: tokenizer HuggingFace
            max_length: longueur max de séquence
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = self._load_data()

    def _load_data(self) -> List[Dict]:
        samples = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw_sample = json.loads(line)
                formatted = format_sample(raw_sample)
                samples.append(formatted)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        prompt = sample["prompt"]
        response = sample["response"]

        # Texte complet vu par le modèle
        full_text = prompt + response

        # Tokenisation complète
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Longueur du prompt seul (sans tokens spéciaux ajoutés)
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = len(prompt_ids)

        # Construction des labels :
        # -100 sur le prompt
        # vrais labels uniquement sur la réponse
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Sécurité en cas de troncature
        labels = labels[: len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }