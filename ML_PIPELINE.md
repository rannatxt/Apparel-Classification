# Apparel Classification ML Pipeline

This document describes the Machine Learning pipeline built for the Apparel Classification project.

## Architecture

The model is based on **ResNet18**, a lightweight architecture suitable for training and inference on consumer hardware like an RTX 3050. The model is initialized with ImageNet weights (transfer learning) and the final classification head is modified to output exactly 3 classes:
- `Indian`
- `Indo-Western`
- `Western`

## Pipeline Overview

1. **Dataset Loading (`src/dataset.py`)**
   - Loads images from `dataset/train/`, `dataset/val/`, and `dataset/test/`.
   - Applies data augmentation (Random Horizontal Flip, Rotation, Color Jitter) to the training set.
   - Calculates class weights dynamically to handle class imbalance (Indo-Western only has 200 samples vs 1500 for the others).

2. **Model Definition (`src/model.py`)**
   - Uses `torchvision.models.resnet18(weights="IMAGENET1K_V1")`.
   - Modifies `model.fc` to output 3 classes.

3. **Training (`src/train.py`)**
   - Uses AdamW optimizer with learning rate scheduling (`ReduceLROnPlateau`).
   - Uses `CrossEntropyLoss` with the calculated class weights.
   - Supports Mixed Precision (`torch.cuda.amp`) for faster GPU training.
   - Implements Early Stopping (patience=3).
   - Saves the best model checkpoint to `models/best_model.pth`.

4. **Evaluation (`src/evaluate.py`)**
   - Evaluates the test set on the best model checkpoint.
   - Generates Accuracy, Precision, Recall, Macro F1, Weighted F1.
   - Outputs a Classification Report and Confusion Matrix.
   - Saves metrics to `reports/evaluation_metrics.json`.

5. **Inference (`src/inference.py`)**
   - Clean, production-ready inference interface `ApparelClassifier`.
   - Provides a `predict(image_path)` function returning JSON with the prediction, confidence, and top predictions.

## Commands

### 1. Train the model
```bash
py -3 src/train.py
```

### 2. Evaluate the model
```bash
py -3 src/evaluate.py
```

### 3. Run Inference on a single image
```bash
py -3 src/inference.py path/to/image.jpg
```

### 4. Run ML Pipeline Tests
```bash
py -3 -m unittest tests/test_ml.py -v
```

## Expected Inference Output
```json
{
  "prediction": "Indian",
  "confidence": 0.942,
  "top_predictions": [
    {
      "class": "Indian",
      "confidence": 0.942
    },
    {
      "class": "Indo-Western",
      "confidence": 0.041
    },
    {
      "class": "Western",
      "confidence": 0.017
    }
  ]
}
```
