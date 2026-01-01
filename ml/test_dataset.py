from transformers import AutoTokenizer
from ml.dataset import InterrogationDataset


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = InterrogationDataset(
    path="data/dataset.jsonl",
    tokenizer=tokenizer
)

sample = dataset[0]

print("input_ids shape:", sample["input_ids"].shape)
print("labels shape:", sample["labels"].shape)