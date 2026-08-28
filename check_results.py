"""Quick test to check if Grad-CAM and LIME images are in the results"""
import json
from pathlib import Path

print("Checking temp_results for recent predictions...")

temp_results = Path('temp_results')
if not temp_results.exists():
    print("No temp_results folder found!")
else:
    result_files = list(temp_results.glob('*.json'))
    print(f"Found {len(result_files)} result files")
    
    if result_files:
        # Get most recent
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"\nLatest result: {latest.name}")
        
        with open(latest, 'r') as f:
            data = json.load(f)
        
        # Check what's in the result
        print("\n=== PREDICTION DATA ===")
        if 'prediction' in data:
            print(f"Class: {data['prediction'].get('predicted_class')}")
            print(f"Confidence: {data['prediction'].get('confidence')}")
        
        print("\n=== EXPLANATIONS DATA ===")
        if 'explanations' in data:
            expl = data['explanations']
            print(f"Methods available: {expl.get('methods_available', [])}")
            print(f"Has Grad-CAM: {'gradcam_base64' in expl}")
            if 'gradcam_base64' in expl:
                gradcam_len = len(expl['gradcam_base64'])
                print(f"  Grad-CAM length: {gradcam_len} chars")
                print(f"  Starts with: {expl['gradcam_base64'][:50]}")
            
            print(f"Has LIME: {'lime_base64' in expl}")
            if 'lime_base64' in expl:
                lime_len = len(expl['lime_base64'])
                print(f"  LIME length: {lime_len} chars")
                print(f"  Starts with: {expl['lime_base64'][:50]}")
            
            print(f"Has reasoning: {'reasoning' in expl}")
            if 'reasoning' in expl:
                print(f"  Reasoning preview: {expl['reasoning'][:200]}")
            
            print(f"Has hateful_keywords: {'hateful_keywords' in expl}")
            if 'hateful_keywords' in expl:
                print(f"  Keywords: {expl['hateful_keywords']}")
        else:
            print("No explanations found!")
    else:
        print("No result files found - run a prediction first!")

print("\n=== CHECKING FLASK SERVER STATUS ===")
import subprocess
try:
    result = subprocess.run(['netstat', '-an'], capture_output=True, text=True, timeout=2)
    if ':5000' in result.stdout:
        print("✓ Port 5000 appears to be in use (Flask may be running)")
    else:
        print("✗ Port 5000 not found - Flask server may not be running")
except Exception as e:
    print(f"Could not check port: {e}")
