import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SceneDataset, stratified_split
from model import load_model

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "feature_maps")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# part a
def get_layer_names():
    return ["stage1", "stage2", "stage3", "stage4"]


# part b
class HookCapture:
    def __init__(self, model, names):
        self.maps = {}
        self.handles = []
        for n, m in model.named_modules():
            if n in names:
                h = m.register_forward_hook(self._hook_fn(n))
                self.handles.append(h)

    def _hook_fn(self, name):
        def hook(mod, inp, out):
            self.maps[name] = out.detach().cpu()
        return hook

    def clear(self):
        self.maps.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# part c
def visualize(model, loader, layers, out_dir, n_imgs=10, ch=0):
    os.makedirs(out_dir, exist_ok=True)

    cap = HookCapture(model, layers)
    model.eval()

    # collect feature maps and original images
    fmaps = {name: [] for name in layers}
    orig_imgs = []

    done = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"]
            for i in range(imgs.size(0)):
                if done >= n_imgs:
                    break

                single = imgs[i].unsqueeze(0).to(device)
                cap.clear()
                model(single)

                # save original image (CHW -> HWC for plotting)
                orig_imgs.append(imgs[i].permute(1, 2, 0).numpy())

                for name in layers:
                    fm = cap.maps[name].squeeze(0)  # (C, H, W)
                    fmaps[name].append(fm[ch].numpy())

                done += 1
            if done >= n_imgs:
                break

    cap.remove()

    # plot
    n_rows = 1 + len(layers)
    n_cols = done
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for c in range(n_cols):
        ax = axes[0, c]
        ax.imshow(np.clip(orig_imgs[c], 0, 1))
        ax.axis("off")
        ax.set_title(f"img {c}", fontsize=8)
    axes[0, 0].set_ylabel("original", fontsize=9, rotation=90, labelpad=10)
    axes[0, 0].yaxis.set_visible(True)
    axes[0, 0].tick_params(left=False, labelleft=False)

    for r, name in enumerate(layers):
        for c in range(n_cols):
            ax = axes[r + 1, c]
            ax.imshow(fmaps[name][c], cmap="viridis")
            ax.axis("off")
        axes[r + 1, 0].set_ylabel(name, fontsize=9, rotation=90, labelpad=10)
        axes[r + 1, 0].yaxis.set_visible(True)
        axes[r + 1, 0].tick_params(left=False, labelleft=False)

    fig.suptitle("Feature maps")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_maps_grid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# part f
def non_positive_stats(model, loader, mod_names, n_imgs=200):
    cap = HookCapture(model, mod_names)
    model.eval()

    # running mean and count per module
    m = {name: 0.0 for name in mod_names}
    n = {name: 0 for name in mod_names}

    done = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"]
            for i in range(imgs.size(0)):
                if done >= n_imgs:
                    break

                single = imgs[i].unsqueeze(0).to(device)
                cap.clear()
                model(single)

                for name in mod_names:
                    fm = cap.maps[name]
                    u = (fm <= 0).float().mean().item()
                    m[name] = (m[name] * n[name] + u) / (n[name] + 1)
                    n[name] += 1

                done += 1
            if done >= n_imgs:
                break

    cap.remove()

    print('module avg % non-positive')
    results = {}
    for name in mod_names:
        pct = m[name] * 100.0
        results[name] = pct
        print(f"{name} {pct}%")
    return results


def main():
    print(f"device: {device}")

    # load model
    model_path = os.path.join(MODELS_DIR, "model1.pth")
    model = load_model(model_path, num_layers=18, pretrained=False)
    model = model.to(device)
    model.eval()
    print(f"loaded {model_path}")

    # make dataloader from val split
    _, val_split, _ = stratified_split(DATA_ROOT, seed=SEED)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_set = SceneDataset(val_split[0], val_split[1], transform=tf)
    loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)

    # part a
    layers = get_layer_names()
    print(f"layers: {layers}")

    # part c
    print("generating feature maps")
    visualize(model, loader, layers, OUT_DIR, n_imgs=10)

    # part e
    five = ["conv1", "stage1", "stage2", "stage3", "stage4"]
    print(f"five modules: {five}")

    # part f
    print("computing stats")
    non_positive_stats(model, loader, five, n_imgs=200)

    print("done")


if __name__ == "__main__":
    main()
