import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import BertTextClassificationDataset
from evaluate import evaluate
from utils import load_config, resolve_project_path, save_curves, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/bert_finetune_config.yaml",
        help="Path to config file relative to version_b_bert_finetune.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    cache_dir = resolve_project_path(config.get("cache_dir", "hf_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    if config.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", str(config["hf_endpoint"]))

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import is_torch_available
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install transformers accelerate"
        ) from exc

    if not is_torch_available():
        raise SystemExit(
            "Transformers did not detect a compatible PyTorch backend. "
            "For torch 2.0.x, run: "
            "python -m pip install \"transformers==4.30.2\" accelerate. "
            "Or upgrade PyTorch and keep a newer Transformers version."
        )

    set_seed(int(config.get("seed", 42)))

    train_csv = resolve_project_path(config["train_csv"])
    valid_csv = resolve_project_path(config["valid_csv"])
    output_dir = resolve_project_path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=int(config["num_classes"]),
            cache_dir=cache_dir,
        ).to(device)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the BERT checkpoint. Check your network/proxy, "
            "or download bert-base-chinese in advance and set model_name to the local path. "
            "If huggingface.co is blocked, set hf_endpoint in the config, for example: "
            "hf_endpoint: https://hf-mirror.com"
        ) from exc

    train_dataset = BertTextClassificationDataset(train_csv, tokenizer, int(config["max_len"]))
    valid_dataset = BertTextClassificationDataset(valid_csv, tokenizer, int(config["max_len"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    optimizer = AdamW(
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
            labels = batch.pop("labels").to(device)
            batch.pop("text")
            model_inputs = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**model_inputs, labels=labels)
            loss = outputs.loss

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
            best_model_dir = output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_model_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(best_model_dir))

        print(
            f"Epoch [{epoch + 1}/{config['epochs']}], "
            f"Loss: {avg_loss:.4f}, Valid Acc: {metrics['accuracy']:.4f}"
        )

    save_json(output_dir / "history.json", history)
    save_curves(output_dir, history)
    save_json(output_dir / "metrics.json", {"best_valid_acc": best_acc})


if __name__ == "__main__":
    main()
