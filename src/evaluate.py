import torch
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.dataset import get_dataloaders
from src.model import get_model

def evaluate_model(model_path='models/best_model.pth'):
    print("Loading test dataset...")
    _, _, test_loader, classes, _ = get_dataloaders(batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    
    model = get_model(num_classes=len(classes))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"WARNING: No weights found at {model_path}. Evaluating untrained model.")
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print nice confusion matrix
    print(f"{'':>15}", end="")
    for c in classes:
        print(f"{c[:8]:>10}", end="")
    print()
    
    for i, row in enumerate(cm):
        print(f"{classes[i][:12]:>15}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()
        
    # Save metrics to JSON for website integration later
    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    os.makedirs('reports', exist_ok=True)
    with open('reports/evaluation_metrics.json', 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    print("\nSaved metrics to reports/evaluation_metrics.json")
    
if __name__ == "__main__":
    evaluate_model()
