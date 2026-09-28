import torch
import torch.nn.functional as F
from PIL import Image
import json
import os
from torchvision import transforms
from src.model import get_model

class ApparelClassifier:
    def __init__(self, model_path='models/best_model.pth', class_map_path='models/class_idx_map.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load class map
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"Class map not found at {class_map_path}. Train the model first.")
            
        with open(class_map_path, 'r') as f:
            idx_to_class = json.load(f)
            # JSON keys are strings, convert to int
            self.idx_to_class = {int(k): v for k, v in idx_to_class.items()}
            self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
            
        # Initialize model
        self.model = get_model(num_classes=len(self.classes), pretrained=False)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"WARNING: Model weights not found at {model_path}. Using random initialization.")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Validation transforms (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def predict(self, image_path):
        """
        Predict the class of an image.
        Returns: { 'prediction': str, 'confidence': float, 'top_predictions': list }
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return {"error": f"Failed to load image: {str(e)}"}
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            
        # Get top predictions
        probs, indices = torch.sort(probabilities, descending=True)
        
        top_predictions = []
        for i in range(len(probs)):
            top_predictions.append({
                "class": self.idx_to_class[indices[i].item()],
                "confidence": float(probs[i].item())
            })
            
        return {
            "prediction": top_predictions[0]["class"],
            "confidence": top_predictions[0]["confidence"],
            "top_predictions": top_predictions
        }

if __name__ == "__main__":
    # Test on a dummy image if one exists
    import sys
    if len(sys.argv) > 1:
        classifier = ApparelClassifier()
        print(json.dumps(classifier.predict(sys.argv[1]), indent=2))
    else:
        print("Usage: py -3 src/inference.py path/to/image.jpg")
