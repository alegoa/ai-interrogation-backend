import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
LORA_PATH = "checkpoints/lora-mistral/epoch_3"

# Chargement GLOBAL (une seule fois)
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()


def build_inference_prompt(suspect, question):
    identity = suspect.identity

    return f"""
Tu incarnes un personnage dans un jeu d’enquête policière.

IDENTITÉ (vérité absolue, ne jamais en dévier) :
- Âge : {identity["age"]}
- Métier : {identity["job"]}
- Résidence : {identity["residence"]}
- Personnalité : {identity["personality"]}

CONTEXTE :
- Alibi : {suspect.alibi}
- Stratégie : {suspect.strategy}
- Peur : {suspect.state["fear"]}
- Pression : {suspect.state["pressure"]}

RÈGLES :
- Réponds uniquement à la question.
- Ne pose jamais de question en retour.
- Ne change jamais ton identité.
- Réponse courte (1 à 3 phrases).

Question :
{question}

Réponse :
""".strip()


def generate_answer(
    suspect, question
) -> str:
    prompt = build_inference_prompt(suspect, question)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    # Sécurité anti-fuite format entraînement
    if "Question :" in text:
        text = text.split("Question :")[0].strip()

    # Coupe à la dernière phrase complète (sécurité jeu)
    for sep in [".", "?", "!"]:
        if sep in text:
            text = text.rsplit(sep, 1)[0] + sep
    
   

    return text.strip()