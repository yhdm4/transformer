from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TextClassificationDataset
from utils import confusion_matrix, save_confusion_matrix_plot, save_error_cases


def evaluate(model, dataset, batch_size, device, output_dir=None, num_classes=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_texts = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            preds = logits.argmax(dim=1)

            all_texts.extend(batch["text"])
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    total = max(1, len(all_labels))
    accuracy = sum(int(a == b) for a, b in zip(all_labels, all_preds)) / total
    matrix = confusion_matrix(all_labels, all_preds, num_classes)

    if output_dir is not None:
        output_dir = Path(output_dir)
        save_error_cases(output_dir / "error_cases.csv", all_texts, all_labels, all_preds)
        save_confusion_matrix_plot(output_dir, matrix)

    return {"accuracy": accuracy, "confusion_matrix": matrix.tolist()}


def build_dataset(csv_path, tokenizer, max_len):
    return TextClassificationDataset(csv_path, tokenizer, max_len)
