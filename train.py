import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from tokenizer import CharTokenizer
from dataset import TextClassificationDataset
from tiny_transformer import TinyTransformerClassifier


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = "data/train.csv"
    valid_csv = "data/valid.csv"

    train_df = pd.read_csv(train_csv)
    texts = train_df["text"].astype(str).tolist()

    tokenizer = CharTokenizer(texts)

    max_len = 64
    batch_size = 32

    train_dataset = TextClassificationDataset(train_csv, tokenizer, max_len)
    valid_dataset = TextClassificationDataset(valid_csv, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = TinyTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=2,
        num_classes=2,
        max_len=max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    num_epochs = 200

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, valid_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Valid Acc: {acc:.4f}")


if __name__ == "__main__":
    main()