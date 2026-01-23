from pathlib import Path
from transformers import AutoTokenizer
from ml.dataset import InterrogationDataset

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

dataset = InterrogationDataset(
    data_path=Path("data/processed/train.jsonl"),
    tokenizer=tokenizer,
    max_length=512,
)

sample = dataset[0]
print(sample.keys())
print(sample["input_ids"].shape)
print(sample["labels"][:50])