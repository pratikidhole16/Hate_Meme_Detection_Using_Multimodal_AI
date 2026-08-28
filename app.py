"""
Hateful Memes Detector - Flask Web Application
Modern web app with XAI explainability (Grad-CAM, LIME)
"""

from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
from flask_cors import CORS
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
import io
import base64
import logging
from datetime import datetime
import secrets
import uuid

from src.preprocessing.ocr_and_annotations import OCREngine

# Import prediction and explanation modules
from predict import MemeClassifier
from explainability import ExplainabilityEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # For session management
CORS(app)

# Add Python built-ins to Jinja2
app.jinja_env.globals.update(zip=zip, enumerate=enumerate)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['RESULTS_FOLDER'] = Path('results')
app.config['TEMP_RESULTS'] = Path('temp_results')  # File-based result storage

# Create folders
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)
app.config['TEMP_RESULTS'].mkdir(exist_ok=True)

# Initialize classifier and explainability engine
logger.info("Loading model...")
classifier = MemeClassifier("saved_models/model_best.pt")
explainer = ExplainabilityEngine(classifier)
logger.info("✅ Model loaded successfully")

# Cache OCR engines so we don't re-initialize heavy readers per request
_ocr_engines = {}


def _get_or_create_engine(name: str) -> OCREngine:
    if name not in _ocr_engines:
        _ocr_engines[name] = OCREngine(name)
    return _ocr_engines[name]


def extract_text_with_ocr(image_path: Path):
    """Run OCR with fallback between EasyOCR and Tesseract."""
    for engine_name in ("easyocr", "tesseract"):
        try:
            engine = _get_or_create_engine(engine_name)
            text = (engine.read(image_path) or "").strip()
            if text:
                return text, engine_name
        except Exception as ocr_err:
            logger.warning("%s OCR failed: %s", engine_name, ocr_err)
    return "", ""


def should_run_ocr(flag_value: str, provided_text: str) -> bool:
    flag = (flag_value or "auto").lower()
    if flag in {"true", "1", "yes", "force", "on"}:
        return True
    if flag == "auto":
        return len((provided_text or "").strip()) == 0
    return False

# Store prediction history in memory (can be extended to DB)
prediction_history = []


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/results')
def results_page():
    """Results page - shows analysis results"""
    result_id = session.get('latest_result_id')
    if not result_id:
        return redirect(url_for('index'))
    
    # Load result from file instead of session
    result_file = app.config['TEMP_RESULTS'] / f"{result_id}.json"
    if not result_file.exists():
        return redirect(url_for('index'))
    
    with open(result_file, 'r') as f:
        result_data = json.load(f)
    return render_template('results.html', result=result_data)


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        text = request.form.get('text', '').strip()
        explain = request.form.get('explain', 'false').lower() == 'true'
        ocr_flag = request.form.get('ocr', 'auto')
        
        if not file or file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file
        upload_path = app.config['UPLOAD_FOLDER'] / file.filename
        file.save(upload_path)
        
        # Auto OCR if requested or text missing
        ocr_text = ""
        ocr_engine = ""
        if should_run_ocr(ocr_flag, text):
            ocr_text, ocr_engine = extract_text_with_ocr(upload_path)

        text_used = text or ocr_text

        # Get prediction
        prediction = classifier.predict_single(str(upload_path), text_used)
        
        if 'error' in prediction:
            return jsonify(prediction), 500
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'image': file.filename,
            'text': text_used,
            'prediction': prediction,
            'ocr_text': ocr_text,
            'ocr_engine': ocr_engine,
            'input_text': text,
        }
        
        # Add explainability if requested
        if explain:
            explanations = explainer.explain_prediction(
                str(upload_path),
                text_used,
                prediction['predicted_class']
            )
            result['explanations'] = explanations
            result['gradcam_image'] = explanations.get('gradcam_base64')
            result['lime_image'] = explanations.get('lime_base64')
            result['feature_importance'] = explanations.get('feature_importance')
            result['attention_regions'] = explanations.get('attention_regions')
        
        # Store in history
        prediction_history.append(result)
        
        # Store result in file instead of session (avoids 4KB cookie limit)
        result_id = str(uuid.uuid4())
        result_file = app.config['TEMP_RESULTS'] / f"{result_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f)
        
        # Store only the result ID in session (tiny)
        session['latest_result_id'] = result_id
        
        # Add result_id to response for client-side use
        result['result_id'] = result_id
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        file = request.files['file']
        import csv
        
        results = []
        errors = []
        
        # Read CSV and process each row
        stream = io.StringIO(file.stream.read().decode('utf8'), newline=None)
        csv_data = csv.DictReader(stream)
        
        for idx, row in enumerate(csv_data, 1):
            try:
                image_path = Path('img') / row.get('image', '')
                text = (row.get('text', '') or '').strip()
                
                if not image_path.exists():
                    errors.append(f"Row {idx}: Image not found - {image_path}")
                    continue
                
                ocr_text = ""
                ocr_engine = ""
                if should_run_ocr("auto", text):
                    ocr_text, ocr_engine = extract_text_with_ocr(image_path)

                text_used = text or ocr_text

                pred = classifier.predict_single(str(image_path), text_used)
                pred['ocr_text'] = ocr_text
                pred['ocr_engine'] = ocr_engine
                pred['text_used'] = text_used
                pred['row'] = idx
                results.append(pred)
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
        
        return jsonify({
            'total': idx,
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """Detailed explanation endpoint"""
    try:
        data = request.json
        image_path = data.get('image_path')
        text = data.get('text')
        predicted_class = data.get('predicted_class')
        
        if not image_path or not text:
            return jsonify({'error': 'Missing image_path or text'}), 400
        
        explanations = explainer.explain_prediction(
            image_path,
            text,
            predicted_class
        )
        
        return jsonify(explanations), 200
    
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def history():
    """Get prediction history"""
    limit = request.args.get('limit', 50, type=int)
    return jsonify(prediction_history[-limit:]), 200


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    global prediction_history
    prediction_history = []
    return jsonify({'message': 'History cleared'}), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get statistics"""
    total = len(prediction_history)
    hateful = sum(1 for p in prediction_history if p['prediction']['predicted_class'] == 1)
    non_hateful = total - hateful
    avg_confidence = np.mean([p['prediction']['confidence'] for p in prediction_history]) if prediction_history else 0
    
    return jsonify({
        'total_predictions': total,
        'hateful_count': hateful,
        'non_hateful_count': non_hateful,
        'hateful_percentage': (hateful / total * 100) if total > 0 else 0,
        'avg_confidence': float(avg_confidence)
    }), 200


@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'Hateful Memes Detector v1.0',
        'accuracy': '70.22%',
        'features': ['prediction', 'gradcam', 'lime', 'batch_processing', 'history']
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Disable reloader to prevent site-packages file touches (e.g., transformers cache) from restarting the server mid-request
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
