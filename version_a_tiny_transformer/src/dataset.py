import pandas as pd
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        required = {"text", "label"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_ids = self.tokenizer.encode(row["text"], self.max_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "text": str(row["text"]),
        }
