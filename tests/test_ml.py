import unittest
import torch
import os
import json
from PIL import Image
from src.model import get_model
from src.dataset import get_transforms
from src.inference import ApparelClassifier

class TestMLPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a dummy image
        cls.dummy_image_path = "dummy_test_img.jpg"
        img = Image.new('RGB', (300, 300), color = 'red')
        img.save(cls.dummy_image_path)
        
        # Create a dummy class map
        os.makedirs('models', exist_ok=True)
        cls.dummy_map_path = "models/test_class_map.json"
        with open(cls.dummy_map_path, 'w') as f:
            json.dump({"0": "Indian", "1": "Indo-Western", "2": "Western"}, f)
            
        # Create a dummy model checkpoint
        cls.dummy_model_path = "models/test_model.pth"
        model = get_model(num_classes=3, pretrained=False)
        torch.save(model.state_dict(), cls.dummy_model_path)
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_image_path):
            os.remove(cls.dummy_image_path)
        if os.path.exists(cls.dummy_map_path):
            os.remove(cls.dummy_map_path)
        if os.path.exists(cls.dummy_model_path):
            os.remove(cls.dummy_model_path)
            
    def test_model_loading(self):
        model = get_model(num_classes=3, pretrained=False)
        self.assertEqual(model.fc.out_features, 3)
        
    def test_checkpoint_loading(self):
        # Test loading the checkpoint
        classifier = ApparelClassifier(
            model_path=self.dummy_model_path,
            class_map_path=self.dummy_map_path
        )
        self.assertIsNotNone(classifier.model)
        
    def test_preprocessing(self):
        train_t, val_t = get_transforms()
        img = Image.open(self.dummy_image_path)
        
        # Check output tensor shape
        out_tensor = val_t(img)
        self.assertEqual(out_tensor.shape, (3, 224, 224))
        
    def test_class_mapping(self):
        classifier = ApparelClassifier(
            model_path=self.dummy_model_path,
            class_map_path=self.dummy_map_path
        )
        self.assertEqual(classifier.classes, ["Indian", "Indo-Western", "Western"])
        
    def test_inference(self):
        classifier = ApparelClassifier(
            model_path=self.dummy_model_path,
            class_map_path=self.dummy_map_path
        )
        result = classifier.predict(self.dummy_image_path)
        
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("top_predictions", result)
        self.assertEqual(len(result["top_predictions"]), 3)
        
    def test_invalid_image(self):
        classifier = ApparelClassifier(
            model_path=self.dummy_model_path,
            class_map_path=self.dummy_map_path
        )
        result = classifier.predict("nonexistent_image_12345.jpg")
        self.assertIn("error", result)

if __name__ == '__main__':
    unittest.main()
