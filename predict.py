import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from torchvision import transforms
from torchvision.io import read_image
import json
import sys

sys.path.insert(0, str(Path.cwd()))
from src.training.train_improved import DualEncoderFusion, TrainConfig

class MemeClassifier:
    """Hateful Memes Classifier for Inference"""
    
    def __init__(self, model_path: str = "saved_models/model_best.pt", device: str = None):
        """Initialize classifier with trained model
        
        Args:
            model_path: Path to saved model checkpoint
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {self.device}")
        
        # Load config
        self.cfg = TrainConfig(annotations=Path("data/annotations.csv"), image_root=Path("img"))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model)
        
        # Load model
        self.model = DualEncoderFusion(self.cfg).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        
        # Class labels
        self.class_names = {0: "Non-Hateful", 1: "Hateful"}
        
        print("✅ Model loaded successfully")
    
    def predict_single(self, image_path: str, text: str, return_confidence: bool = True):
        """Predict class for a single image-text pair
        
        Args:
            image_path: Path to image file
            text: Text/caption for the meme
            return_confidence: Whether to return confidence scores
            
        Returns:
            dict with prediction, confidence, and class label
        """
        try:
            # Load and preprocess image
            image = read_image(str(image_path)).float() / 255.0
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            elif image.shape[0] > 3:
                image = image[:3]
            
            # Resize
            transform = transforms.Resize((224, 224))
            image = transform(image)
            image = image.unsqueeze(0).to(self.device)
            
            # Tokenize text
            tokens = self.tokenizer(
                [text[:200]],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Inference
            with torch.no_grad():
                logits = self.model(image, tokens)
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
            
            result = {
                "predicted_class": pred_class,
                "class_name": self.class_names[pred_class],
                "confidence": float(confidence),
            }
            
            if return_confidence:
                result["class_probabilities"] = {
                    self.class_names[i]: float(probs[0, i].item())
                    for i in range(len(self.class_names))
                }
            
            return result
        
        except Exception as e:
            return {"error": str(e), "image_path": image_path}
    
    def predict_batch(self, items: list, return_confidence: bool = True):
        """Predict for multiple image-text pairs
        
        Args:
            items: List of dicts with 'image' and 'text' keys
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of predictions
        """
        results = []
        for item in items:
            result = self.predict_single(item["image"], item["text"], return_confidence)
            result["image"] = item["image"]
            result["text"] = item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"]
            results.append(result)
        return results
    
    def explain_prediction(self, prediction: dict):
        """Provide human-readable explanation of prediction
        
        Args:
            prediction: Result from predict_single or predict_batch
            
        Returns:
            Formatted explanation string
        """
        if "error" in prediction:
            return f"❌ Error: {prediction['error']}"
        
        class_name = prediction["class_name"]
        confidence = prediction["confidence"]
        
        if confidence < 0.55:
            certainty = "LOW (model uncertain)"
        elif confidence < 0.65:
            certainty = "MEDIUM"
        else:
            certainty = "HIGH"
        
        explanation = f"""
        Class: {class_name}
        Confidence: {confidence*100:.2f}%
        Certainty: {certainty}
        """
        
        if "class_probabilities" in prediction:
            explanation += "\n        Class Probabilities:"
            for cls, prob in prediction["class_probabilities"].items():
                explanation += f"\n          {cls}: {prob*100:.2f}%"
        
        return explanation


def main():
    """Example usage"""
    
    # Initialize classifier
    classifier = MemeClassifier("saved_models/model_best.pt")
    
    # Example 1: Single prediction
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*70)
    
    # Try to find a sample image
    img_dir = Path("img")
    if img_dir.exists():
        sample_images = list(img_dir.glob("*.png"))[:1]
        if sample_images:
            img_path = sample_images[0]
            text = "This is a sample meme"
            
            print(f"Image: {img_path.name}")
            print(f"Text: {text}")
            
            result = classifier.predict_single(str(img_path), text)
            print(f"\nPrediction:")
            print(classifier.explain_prediction(result))
        else:
            print("No sample images found in img/ directory")
    
    # Example 2: Batch prediction
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Prediction (If images available)")
    print("="*70)
    
    if img_dir.exists():
        sample_images = list(img_dir.glob("*.png"))[:3]
        if sample_images:
            batch_items = [
                {"image": str(img), "text": "Sample text"}
                for img in sample_images
            ]
            
            results = classifier.predict_batch(batch_items)
            print(f"\nBatch Results ({len(results)} items):")
            for i, result in enumerate(results, 1):
                print(f"\n  Item {i}: {result.get('image', 'N/A')}")
                print(f"    Prediction: {result['class_name']}")
                print(f"    Confidence: {result['confidence']*100:.2f}%")
    
    print("\n" + "="*70)
    print("USAGE IN YOUR CODE:")
    print("="*70)
    print("""
from predict import MemeClassifier

# Initialize
classifier = MemeClassifier("saved_models/model_best.pt")

# Single prediction
result = classifier.predict_single("path/to/image.png", "Meme text here")
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']*100:.2f}%")

# Batch prediction
items = [
    {"image": "img1.png", "text": "text1"},
    {"image": "img2.png", "text": "text2"},
]
results = classifier.predict_batch(items)

# Get explanation
explanation = classifier.explain_prediction(result)
print(explanation)
    """)


if __name__ == "__main__":
    main()
