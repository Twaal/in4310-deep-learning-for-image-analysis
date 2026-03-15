import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from dataset import SceneDataset, stratified_split, verify_disjoint, CLASSES
from model import get_resnet, save_model

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_transforms(augment=False):
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def build_optimizer(model, cfg):
    if cfg["training"]["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    elif cfg["training"]["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["training"]["lr"], momentum=cfg["training"].get("momentum", 0.9), weight_decay=cfg["training"]["weight_decay"])


def compute_metrics(model, loader, criterion=None):
    model.eval()
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    num_classes = len(CLASSES)
    ap_per_class = []
    acc_per_class = []
    for c in range(num_classes):
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
    val_loss = running_loss / total if criterion else None
    return mAP, mean_acc, ap_per_class, acc_per_class, val_loss


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch in loader:
        imgs = batch["img"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def plot_curves(train_losses, val_losses, test_losses, val_maps, val_accs, test_maps, test_accs, save_dir, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_losses, label="train")
    axes[0].plot(val_losses, label="val")
    axes[0].plot(test_losses, label="test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(val_maps, label="val")
    axes[1].plot(test_maps, label="test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()
    axes[1].set_title("mAP")
    axes[2].plot(val_accs, label="val")
    axes[2].plot(test_accs, label="test")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean Acc")
    axes[2].legend()
    axes[2].set_title("Mean Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_curves.png"))
    plt.close()


def train_with_config(cfg_path, train_split, val_split, test_split):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\nStarting {os.path.basename(cfg_path)}")

    train_tf, val_tf = get_transforms(augment=cfg["augmentation"])
    train_set = SceneDataset(train_split[0], train_split[1], transform=train_tf)
    val_set = SceneDataset(val_split[0], val_split[1], transform=val_tf)
    test_set = SceneDataset(test_split[0], test_split[1], transform=val_tf)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=8)

    model = get_resnet(num_layers=cfg["model"]["num_layers"], pretrained=cfg["model"]["pretrained"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # im using cross entropy loss for multi-class classification
    optimizer = build_optimizer(model, cfg)

    save_path = os.path.join(os.path.dirname(__file__), "..", cfg["save_path"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dir = os.path.dirname(save_path)
    model_name = os.path.splitext(os.path.basename(save_path))[0]
    print(f"Final model will be saved to: {save_path}")

    train_losses = []
    val_losses = []
    val_maps = []
    val_accs = []
    test_losses = []
    test_maps = []
    test_accs = []
    best_map = 0.0

    epochs = cfg["training"]["epochs"]
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        mAP, mean_acc, _, _, val_loss = compute_metrics(model, val_loader, criterion)
        t_mAP, t_acc, _, _, t_loss = compute_metrics(model, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maps.append(mAP)
        val_accs.append(mean_acc)
        test_losses.append(t_loss)
        test_maps.append(t_mAP)
        test_accs.append(t_acc)

        print(f"[{epoch+1}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} test_loss={t_loss:.4f} mAP={mAP:.4f} acc={mean_acc:.4f}")

        if mAP > best_map: # only saving the model weights with best val performance on mAP metric
            best_map = mAP
            save_model(model, save_path)
            print(f"Best mAP={best_map:.4f}")

    plot_curves(train_losses, val_losses, test_losses, val_maps, val_accs, test_maps, test_accs, save_dir, model_name)
    print(f"Done, best mAP: {best_map:.4f}")
    return best_map

# train three models with the given config files
if __name__ == "__main__":
    train_split, val_split, test_split = stratified_split(DATA_ROOT, seed=SEED)
    verify_disjoint(train_split[0], val_split[0], test_split[0])

    configs = ["config1.yaml", "config2.yaml", "config3.yaml", "config1_transfer.yaml", "config2_transfer.yaml", "config3_transfer.yaml"]
    for file in configs:
        path = os.path.join(CONFIGS_DIR, file)
        train_with_config(path, train_split, val_split, test_split)

    print("Training done.")
