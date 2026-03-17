import os
import csv
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import average_precision_score

from dataset import SceneDataset, stratified_split, verify_disjoint, CLASSES
from model import load_model

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def evaluate(model, loader, criterion):
    model.eval()
    all_labels = []
    all_probs = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    test_loss = running_loss / total

    ap_per_class = []
    acc_per_class = []
    for c in range(len(CLASSES)):
        binary = (all_labels == c).astype(int)
        ap = average_precision_score(binary, all_probs[:, c])
        ap_per_class.append(ap)

        mask = all_labels == c
        if mask.sum() > 0:
            class_preds = all_probs[mask].argmax(axis=1)
            acc_per_class.append((class_preds == c).mean())
        else:
            acc_per_class.append(0.0)

    mAP = np.mean(ap_per_class)
    mean_acc = np.mean(acc_per_class)
    return mAP, mean_acc, ap_per_class, acc_per_class, test_loss, all_probs


def verify_scores(model_path, num_layers, scores_path, loader):
    model = load_model(model_path, num_layers=num_layers)
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    new_scores = np.concatenate(all_probs)
    saved_scores = np.load(scores_path)

    assert np.allclose(saved_scores, new_scores, atol=1e-6), "Scores do not match!"
    print("Scores match!")


if __name__ == "__main__":
    train_split, val_split, test_split = stratified_split(DATA_ROOT, seed=SEED)
    verify_disjoint(train_split[0], val_split[0], test_split[0])

    val_set = SceneDataset(val_split[0], val_split[1], transform=test_tf)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=8)

    test_set = SceneDataset(test_split[0], test_split[1], transform=test_tf)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()

    csv_path = os.path.join(MODELS_DIR, "metrics.csv")
    csv_rows = []

    configs = ["config1.yaml", "config2.yaml", "config3.yaml"]
    for cfg_file in configs:
        cfg_path = os.path.join(CONFIGS_DIR, cfg_file)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        model_path = os.path.join(os.path.dirname(__file__), "..", cfg["save_path"])
        num_layers = cfg["model"]["num_layers"]
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        model = load_model(model_path, num_layers=num_layers)
        model = model.to(device)

        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            mAP, mean_acc, ap_per_class, acc_per_class, loss, scores = evaluate(model, loader, criterion)

            print(f"\n==={model_name} / {split_name}===")
            print(f"Loss: {loss:.4f}  mAP: {mAP:.4f}  Mean Acc: {mean_acc:.4f}")
            for i, cls in enumerate(CLASSES):
                print(f"{cls}  AP={ap_per_class[i]:.4f}  Acc={acc_per_class[i]:.4f}")

            row = {
                "model": model_name,
                "split": split_name,
                "loss": round(loss, 4),
                "mAP": round(mAP, 4),
                "mean_acc": round(mean_acc, 4),
            }
            for i, cls in enumerate(CLASSES):
                row[f"AP_{cls}"] = round(ap_per_class[i], 4)
                row[f"Acc_{cls}"] = round(acc_per_class[i], 4)
            csv_rows.append(row)

            if split_name == "test":
                scores_path = os.path.join(MODELS_DIR, f"{model_name}_scores.npy")
                np.save(scores_path, scores)
                print(f"Saved scores to {scores_path}")
                verify_scores(model_path, num_layers, scores_path, test_loader)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nMetrics saved to {csv_path}")
