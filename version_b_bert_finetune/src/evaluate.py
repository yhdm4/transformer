import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import BertTextClassificationDataset
from utils import (
    confusion_matrix,
    load_config,
    resolve_project_path,
    save_confusion_matrix_plot,
    save_error_cases,
    save_json,
)


def evaluate(model, dataset, batch_size, device, output_dir=None, num_classes=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_texts = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            texts = batch.pop("text")
            model_inputs = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**model_inputs)
            preds = outputs.logits.argmax(dim=1)

            all_texts.extend(texts)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    total = max(1, len(all_labels))
    accuracy = sum(int(a == b) for a, b in zip(all_labels, all_preds)) / total
    matrix = confusion_matrix(all_labels, all_preds, num_classes)

    if output_dir is not None:
        save_error_cases(output_dir / "error_cases.csv", all_texts, all_labels, all_preds)
        save_confusion_matrix_plot(output_dir, matrix)

    return {"accuracy": accuracy, "confusion_matrix": matrix.tolist()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/bert_finetune_config.yaml",
        help="Path to config file relative to version_b_bert_finetune.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Model checkpoint directory. Defaults to output_dir/best_model.",
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

    output_dir = resolve_project_path(config.get("output_dir", "results"))
    checkpoint = args.checkpoint or str(output_dir / "best_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            cache_dir=cache_dir,
        ).to(device)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the checkpoint. Check that best_model exists, "
            "or pass --checkpoint with a valid local model directory."
        ) from exc

    valid_dataset = BertTextClassificationDataset(
        resolve_project_path(config["valid_csv"]),
        tokenizer,
        int(config["max_len"]),
    )
    metrics = evaluate(
        model,
        valid_dataset,
        int(config["batch_size"]),
        device,
        output_dir=output_dir,
        num_classes=int(config["num_classes"]),
    )
    save_json(output_dir / "eval_metrics.json", metrics)
    print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
