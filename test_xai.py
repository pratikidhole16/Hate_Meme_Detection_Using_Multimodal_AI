"""Test XAI features to verify Grad-CAM and LIME are working"""
import logging
logging.basicConfig(level=logging.INFO)

print("Testing XAI functionality...")

# Test imports
try:
    from pytorch_grad_cam import GradCAM
    print("✓ Grad-CAM available")
except ImportError as e:
    print(f"✗ Grad-CAM not available: {e}")

try:
    import lime
    import lime.lime_image
    print("✓ LIME available")
except ImportError as e:
    print(f"✗ LIME not available: {e}")

try:
    from transformers import pipeline
    print("✓ Transformers available")
except ImportError as e:
    print(f"✗ Transformers not available: {e}")

# Test if model classifier can be loaded
try:
    from predict import MemeClassifier
    import torch
    
    print("\nLoading classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # This will take a moment
    classifier = MemeClassifier("saved_models/model_best.pt")
    print("✓ Classifier loaded")
    
    # Test explainability
    from explainability import ExplainabilityEngine
    explainer = ExplainabilityEngine(classifier)
    print("✓ ExplainabilityEngine created")
    
    # Find a test image
    import os
    from pathlib import Path
    
    test_img = None
    for img_path in Path('img').glob('*.jpg'):
        test_img = str(img_path)
        break
    
    if test_img:
        print(f"\nTesting with image: {test_img}")
        test_text = "This is a test meme"
        
        # Test prediction
        pred = classifier.predict(test_img, test_text)
        print(f"Prediction: {pred['predicted_class']} (confidence: {pred['confidence']:.2%})")
        
        # Test explanations
        print("\nGenerating explanations...")
        explanations = explainer.explain_prediction(test_img, test_text, pred['predicted_class'])
        
        print(f"Methods available: {explanations.get('methods_available', [])}")
        print(f"Grad-CAM: {'✓' if explanations.get('gradcam_base64') else '✗'}")
        print(f"LIME: {'✓' if explanations.get('lime_base64') else '✗'}")
        print(f"Reasoning: {explanations.get('reasoning', 'N/A')[:100]}...")
        
    else:
        print("No test image found in img/ directory")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
