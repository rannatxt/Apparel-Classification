import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import json

def get_transforms():
    """
    Standard ImageNet transforms.
    Training includes light data augmentation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(data_dir="dataset", batch_size=32, num_workers=0):
    """
    Creates PyTorch DataLoaders for train, val, and test splits.
    Includes class weights for the loss function based on the training distribution.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)
    
    # Save class_to_idx for inference mapping
    os.makedirs('models', exist_ok=True)
    with open('models/class_idx_map.json', 'w') as f:
        # Save idx -> class_name
        idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        json.dump(idx_to_class, f)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Calculate class weights for CrossEntropyLoss to handle Indo-Western imbalance
    # Weight = total_samples / (num_classes * class_samples)
    from collections import Counter
    class_counts = Counter(train_dataset.targets)
    total_samples = len(train_dataset)
    num_classes = len(train_dataset.classes)
    
    class_weights = []
    for i in range(num_classes):
        w = total_samples / (num_classes * class_counts[i])
        class_weights.append(w)
        
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights_tensor
