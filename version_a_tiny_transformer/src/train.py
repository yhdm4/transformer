import argparse

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TextClassificationDataset
from evaluate import evaluate
from model import TinyTransformerClassifier
from tokenizer import CharTokenizer
from utils import load_config, resolve_project_path, save_curves, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/tiny_transformer_config.yaml",
        help="Path to config file relative to version_a_tiny_transformer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    set_seed(int(config.get("seed", 42)))

    train_csv = resolve_project_path(config["train_csv"])
    valid_csv = resolve_project_path(config["valid_csv"])
    output_dir = resolve_project_path(config.get("output_dir", "../results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_csv)
    tokenizer = CharTokenizer(train_df["text"].astype(str).tolist(), int(config["min_freq"]))

    train_dataset = TextClassificationDataset(train_csv, tokenizer, int(config["max_len"]))
    valid_dataset = TextClassificationDataset(valid_csv, tokenizer, int(config["max_len"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    model = TinyTransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        dim=int(config["dim"]),
        num_heads=int(config["num_heads"]),
        ff_dim=int(config["ff_dim"]),
        num_layers=int(config["num_layers"]),
        num_classes=int(config["num_classes"]),
        max_len=int(config["max_len"]),
        pad_id=tokenizer.pad_id,
        dropout=float(config["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history = {"train_loss": [], "valid_acc": []}
    best_acc = -1.0

    for epoch in range(int(config["epochs"])):
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

        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate(
            model,
            valid_dataset,
            int(config["batch_size"]),
            device,
            output_dir=output_dir,
            num_classes=int(config["num_classes"]),
        )

        history["train_loss"].append(avg_loss)
        history["valid_acc"].append(metrics["accuracy"])

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "stoi": tokenizer.stoi,
            }
            with (output_dir / "best_model.pt").open("wb") as f:
                torch.save(checkpoint, f)

        print(
            f"Epoch [{epoch + 1}/{config['epochs']}], "
            f"Loss: {avg_loss:.4f}, Valid Acc: {metrics['accuracy']:.4f}"
        )

    save_json(output_dir / "history.json", history)
    save_curves(output_dir, history)
    save_json(output_dir / "metrics.json", {"best_valid_acc": best_acc})


if __name__ == "__main__":
    main()
