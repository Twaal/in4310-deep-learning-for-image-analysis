import os
import pickle

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights
from torchvision.transforms import transforms

features = {}


def get_features(name):
    def hook(model, x, output):
        features[name] = output
    return hook


class ResNet101Encoder(nn.Module):
    def __init__(self):
        super(ResNet101Encoder, self).__init__()
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        resnet.layer4.register_forward_hook(get_features('layer4'))
        self.encoder = resnet
        self.encoder.eval()
        self.dim = 2048

    def forward(self, x):
        self.encoder(x)
        return features['layer4']


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        resnet.layer4.register_forward_hook(get_features('layer4'))
        self.encoder = resnet
        self.encoder.eval()
        self.dim = 2048

    def forward(self, x):
        self.encoder(x)
        return features['layer4']


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.layer4.register_forward_hook(get_features('layer4'))
        self.encoder = resnet
        self.encoder.eval()
        self.dim = 512

    def forward(self, x):
        self.encoder(x)
        return features['layer4']


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_names = [i for i in os.listdir(img_dir) if i.endswith('jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, img_name.split('.')[0]


def coco_collate_fn(batch):
    # Separate images, labels, and filenames
    images = [item[0] for item in batch]
    filenames = [item[1] for item in batch]
    # Collate images and labels into tensors (assuming they are tensors)
    images = torch.stack(images).contiguous()

    return images, filenames


def extract_coco_features(images_dir, save_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # Initialize dataset and dataloader
    dataset = CocoDataset(images_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10, pin_memory=True,
                                             collate_fn=coco_collate_fn)

    # Initialize model and move to GPU if available
    device = torch.device('cuda', 0)
    model = ResNet18Encoder().to(device)

    coc_features_dict = {}
    with torch.no_grad():
        for images, img_ids in dataloader:
            images = images.to(device)
            features = model(images).cpu().numpy()
            for img_id, feat in zip(img_ids, features):
                coc_features_dict[img_id] = feat

    # Save features to a pickle file
    with open(save_path, 'wb') as f:
        pickle.dump(coc_features_dict, f)

    print(f'Saved extracted features to {save_path}', flush=True)


if __name__ == '__main__':
    extract_coco_features('/fp/projects01/ec35/data/coco/train2017/', 'coco_train_resnet18_layer4_features.pkl')
    extract_coco_features('/fp/projects01/ec35/data/coco/val2017/', 'coco_val_resnet18_layer4_features.pkl')
