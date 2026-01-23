"""
ml/train.py

Entraînement LoRA explicite (PyTorch) pour Mistral 7B.

- Dataset : InterrogationDataset
- Modèle : Mistral 7B + QLoRA + LoRA
- Optimisation : AdamW
- Loss : Causal LM (gérée par le modèle)
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from ml.dataset import InterrogationDataset
from ml.model import load_model, load_tokenizer


# -----------------------
# Configuration
# -----------------------

TRAIN_PATH = Path("data/dataset_v2newnew.jsonl")
VAL_PATH = Path("data/processed/val.jsonl")

OUTPUT_DIR = Path("checkpoints/lora-mistral")

BATCH_SIZE = 1                     # obligatoire avec 12 Go
GRAD_ACCUM_STEPS = 8               # batch effectif = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

LOG_EVERY = 20
SAVE_EVERY_EPOCH = True


# -----------------------
# Utils
# -----------------------

def collate_fn(batch):
    """
    Padding dynamique pour batch_size=1 (simple et sûr).
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# -----------------------
# Main
# -----------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Chargement du tokenizer et du modèle...")
    global tokenizer
    tokenizer = load_tokenizer()
    model = load_model()
    model.train()

    print("Chargement du dataset...")
    train_dataset = InterrogationDataset(
        data_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    global_step = 0

    print("Début de l'entraînement")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            epoch_loss += loss.item()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % LOG_EVERY == 0:
                    avg_loss = epoch_loss / LOG_EVERY
                    print(f"[Epoch {epoch+1}] Step {global_step} | Loss: {avg_loss:.4f}")
                    epoch_loss = 0.0

        if SAVE_EVERY_EPOCH:
            save_path = OUTPUT_DIR / f"epoch_{epoch+1}"
            model.save_pretrained(save_path)
            print(f"Checkpoint sauvegardé : {save_path}")

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
