"""Test the complete XAI pipeline with a real prediction"""
import requests
import json
from pathlib import Path
import time

BASE_URL = "http://127.0.0.1:5000"

print("="*60)
print("Testing Hateful Memes XAI Pipeline")
print("="*60)

# Find a test image
test_images = list(Path('img').glob('*.jpg'))[:1]

if not test_images:
    print("ERROR: No test images found in img/ directory")
    exit(1)

test_image = test_images[0]
print(f"\n📷 Using test image: {test_image}")

# Test with different texts
test_cases = [
    ("This is a normal meme", "Safe meme"),
    ("hate stupid garbage", "Hateful meme with offensive words"),
]

for text, description in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Text: '{text}'")
    print("="*60)
    
    # Upload and predict
    with open(test_image, 'rb') as img_file:
        files = {
            'image': img_file,
            'text': (None, text)
        }
        
        print("\n📤 Sending prediction request...")
        response = requests.post(f"{BASE_URL}/api/predict", files=files)
        
        if response.status_code != 200:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            continue
        
        result = response.json()
        session_id = result.get('session_id')
        
        if not session_id:
            print("❌ No session_id returned")
            continue
        
        print(f"✅ Prediction successful, session: {session_id}")
        
        # Wait a moment for file to be written
        time.sleep(1)
        
        # Check the results file
        results_file = Path(f'temp_results/{session_id}.json')
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            continue
        
        print(f"✅ Results file found")
        
        with open(results_file, 'r') as f:
            full_result = json.load(f)
        
        # Check prediction
        pred = full_result.get('prediction', {})
        print(f"\n🎯 Prediction: Class {pred.get('predicted_class')} (confidence: {pred.get('confidence', 0):.2%})")
        
        # Check explanations
        expl = full_result.get('explanations', {})
        
        print(f"\n📊 Explanations available:")
        print(f"  - Reasoning: {'✅' if expl.get('reasoning') else '❌'}")
        if expl.get('reasoning'):
            print(f"    Preview: {expl['reasoning'][:150]}...")
        
        print(f"  - Grad-CAM: {'✅' if expl.get('gradcam_base64') else '❌'}")
        if expl.get('gradcam_base64'):
            length = len(expl['gradcam_base64'])
            print(f"    Size: {length:,} chars (base64)")
        
        print(f"  - LIME: {'✅' if expl.get('lime_base64') else '❌'}")
        if expl.get('lime_base64'):
            length = len(expl['lime_base64'])
            print(f"    Size: {length:,} chars (base64)")
        
        print(f"  - Hateful Keywords: {'✅' if expl.get('hateful_keywords') else '❌'}")
        if expl.get('hateful_keywords'):
            print(f"    Keywords: {expl['hateful_keywords']}")
        
        print(f"  - Text Importance: {'✅' if expl.get('text_importance') else '❌'}")
        
        # Display results page URL
        print(f"\n🌐 View full results at: {BASE_URL}/results")
        print(f"   (Session: {session_id})")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
