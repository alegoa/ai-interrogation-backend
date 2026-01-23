"""
ml/model.py

Responsabilité unique :
- Charger le modèle de base (Mistral 7B)
- Appliquer la quantization 4-bit (QLoRA)
- Appliquer la configuration LoRA (PEFT)

Aucune logique d'entraînement.
Aucune logique dataset.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


MODEL_NAME = "mistralai/Mistral-7B-v0.1"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # Mistral n'a pas de pad_token par défaut
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,              # QLoRA
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(model, lora_config)

    # Log utile pour vérifier
    model.print_trainable_parameters()

    return model
