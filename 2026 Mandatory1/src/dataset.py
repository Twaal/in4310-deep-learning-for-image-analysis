import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

class SceneDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"img": img, "labels": self.labels[idx]}


def load_all_filepaths(data_root):
    filepaths, labels = [], []
    for i, cls in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(data_root, cls, "*.jpg"))) # sort to keep the same order
        filepaths.extend(files)
        labels.extend([i] * len(files))

    # convert flattened filepaths to np
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    return filepaths, labels


def stratified_split(data_root, seed=42):
    filepaths, labels = load_all_filepaths(data_root)
    total = len(filepaths)

    rest, test_f, rest_l, test_l = train_test_split(
        filepaths, labels, test_size=3000/total, stratify=labels, random_state=seed
    )
    train_f, val_f, train_l, val_l = train_test_split(
        rest, rest_l, test_size=2000/(total - 3000), stratify=rest_l, random_state=seed
    )
    print(f"Train: {len(train_f)}, Val: {len(val_f)}, Test: {len(test_f)}")
    return (train_f, train_l), (val_f, val_l), (test_f, test_l)

# part b
def verify_disjoint(train_f, val_f, test_f):
    s1, s2, s3 = set(train_f), set(val_f), set(test_f)
    assert not (s1 & s2) and not (s1 & s3) and not (s2 & s3), "Splits overlap!"
    print("Dataset OK")


# make the module runnable to test the split
if __name__ == "__main__":
    # Get the data
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    # SPlit into train, val ,tst
    splits = stratified_split(data_root)

    # Verify the dataset
    verify_disjoint(splits[0][0], splits[1][0], splits[2][0])