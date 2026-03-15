import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet

NUM_CLASSES = 6

def get_resnet(num_layers=18, pretrained=False):
    if pretrained:
        # for part f
        pretrained_models = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        model = pretrained_models[num_layers](pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
    else:
        model = ResNet(img_channels=3, num_layers=num_layers, num_classes=NUM_CLASSES)

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, num_layers=18, pretrained=False):
    model = get_resnet(num_layers=num_layers, pretrained=pretrained)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

if __name__ == "__main__":
    # test saving and loading the model
    model = get_resnet(num_layers=18, pretrained=False)
    save_model(model, "test_model.pth")
    loaded_model = load_model("test_model.pth", num_layers=18, pretrained=False)
    print("Model saved and loaded OK")