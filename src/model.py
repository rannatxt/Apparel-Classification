import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=3, pretrained=True):
    """
    Returns a lightweight ResNet18 model modified for our specific number of classes.
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = resnet18(weights=weights)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Simple check
    model = get_model()
    print("Model initialized.")
    print("Classifier layer:", model.fc)
