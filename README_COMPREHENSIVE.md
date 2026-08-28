# Hateful Memes Detector - Complete Technical Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Architecture](#architecture)
4. [Installation & Setup](#installation--setup)
5. [Application Workflow](#application-workflow)
6. [API Endpoints](#api-endpoints)
7. [Components Explanation](#components-explanation)
8. [Model Details](#model-details)
9. [Explainability & XAI Features](#explainability--xai-features)
10. [Directory Structure](#directory-structure)
11. [Advanced Features](#advanced-features)

---

## Project Overview

**Hateful Memes Detector** is a comprehensive multimodal AI application designed to detect hate speech in image-text memes. The application combines computer vision and natural language processing to identify hateful content with explainability features powered by advanced XAI (Explainable AI) techniques.

### Key Capabilities
- 🖼️ **Image-Text Analysis**: Processes memes as multimodal inputs (images + text)
- 🎯 **Hateful Content Detection**: Binary classification (hateful vs. non-hateful)
- 📊 **Batch Processing**: Analyze multiple memes via CSV upload
- 🔍 **Explainability**: Grad-CAM, LIME, and feature importance visualizations
- 📝 **Automatic OCR**: Text extraction from images when not provided
- 📈 **Analytics Dashboard**: Real-time statistics and prediction history
- 🌐 **REST API**: Full-featured API for integration

---

## Tech Stack

### **Backend Framework**
- **Flask** - Lightweight Python web framework for REST API
- **Flask-CORS** - Cross-Origin Resource Sharing support

### **Deep Learning & ML Libraries**
- **PyTorch** (≥2.1.2) - Deep learning framework
- **TorchVision** (≥0.16.2) - Computer vision utilities
- **Transformers** (≥4.37.0) - Pre-trained NLP models (Hugging Face)
- **TimM** (≥0.9.12) - Vision Transformer implementations
- **Accelerate** (≥0.25.0) - Distributed training utilities

### **Image Processing & OCR**
- **Pillow** (≥10.1.0) - Image processing
- **OpenCV** (≥4.8.1.78) - Computer vision operations
- **EasyOCR** (≥1.7.1) - Optical Character Recognition (primary)
- **Tesseract/PyTesseract** (≥0.3.10) - Fallback OCR engine
- **Albumentations** (≥1.3.1) - Image augmentation

### **Explainability & Interpretability**
- **pytorch-grad-cam** (≥1.4.0) - Gradient-weighted Class Activation Mapping
- **LIME** (≥0.2.0.1) - Local Interpretable Model-agnostic Explanations
- **SHAP** (≥0.44.0) - SHapley Additive exPlanations
- **Captum** (≥0.6.0) - Model interpretability library

### **Data & Analytics**
- **NumPy** (≥1.24.4) - Numerical computing
- **Pandas** (≥2.1.4) - Data manipulation
- **Scikit-learn** (≥1.3.2) - Machine learning utilities
- **Matplotlib** (≥3.8.2) - Plotting & visualization
- **Seaborn** (≥0.13.0) - Statistical visualization

### **Utilities**
- **tqdm** (≥4.66.1) - Progress bars
- **Streamlit** (≥1.29.0) - Alternative UI framework
- **FPDF2** (≥2.7.9) - PDF report generation

---

## Architecture

### **High-Level System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web Application                    │
│                       (app.py)                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼─────┐         ┌────▼────────┐
   │  Predict │         │ Explainability
   │  Module  │         │  Module
   │(predict. │         │(explainability)
   │   py)    │         │
   └────┬─────┘         └────┬────────┘
        │                    │
   ┌────▼────────────────────▼────┐
   │   DualEncoderFusion Model    │
   │  (Vision + Text Fusion)      │
   │  - Vision Encoder (ViT)      │
   │  - Text Encoder (BERT-like)  │
   │  - Fusion Layer              │
   │  - Classification Head       │
   └────┬─────────────────────────┘
        │
   ┌────▼──────────────┐
   │  Feature Layers   │
   │  - Image Features │
   │  - Text Features  │
   │  - Fused Features │
   └───────────────────┘
```

### **Data Flow Architecture**

```
User Upload (Image + Text)
        │
        ▼
┌──────────────────────────┐
│  File Validation         │
│  - Image format check    │
│  - Size validation       │
└───────────┬──────────────┘
            │
            ▼
    ┌────────────────┐
    │  Auto-OCR?     │
    │  (if needed)   │
    └────────┬───────┘
             │
             ▼
   ┌─────────────────────┐
   │  Text Processing    │
   │  - Tokenization     │
   │  - Normalization    │
   └──────────┬──────────┘
              │
              ▼
  ┌───────────────────────┐
  │ Image Preprocessing   │
  │ - Resize to 224×224   │
  │ - Normalize           │
  │ - Channel handling    │
  └───────────┬───────────┘
              │
              ▼
   ┌────────────────────────┐
   │  Model Inference       │
   │  (DualEncoderFusion)   │
   └───────────┬────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   Prediction    Explanations?
   - Class         │
   - Confidence    ├─ Grad-CAM
               │     ├─ LIME
               │     ├─ Feature Importance
               │     └─ Attention Maps
        │
        ▼
   JSON Response
   (with result_id)
```

---

## Installation & Setup

### **Prerequisites**
- Python 3.8+
- pip or conda
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- 8GB+ RAM

### **Step 1: Clone and Navigate**
```bash
cd c:\Users\nangi\Downloads\archive\ (5)\data
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**For GPU Support (Optional but Recommended):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Step 3: Verify Installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### **Step 4: Download Pre-trained Models**
```bash
# Models are automatically downloaded when first used
# Ensure you have ~2GB free disk space for model caches
```

---

## Application Workflow

### **1. Single Image Prediction Workflow**

```
1. User opens http://localhost:5000
2. User uploads image + provides text (or enables OCR)
3. Frontend sends POST to /api/predict
4. Backend:
   a. Validates file
   b. Saves to uploads/
   c. Runs OCR if enabled (EasyOCR → Tesseract fallback)
   d. Preprocesses image & text
   e. Runs inference through DualEncoderFusion
   f. Generates confidence scores
   g. Optional: Generates explanations (Grad-CAM, LIME)
   h. Saves result to temp_results/{result_id}.json
   i. Stores result_id in session
5. Result displayed on /results page
6. Result stored in prediction_history (in-memory)
```

### **2. Batch Processing Workflow**

```
1. User uploads CSV file with columns: image, text
2. POST to /api/batch-predict
3. Backend processes each row:
   a. Load image from img/ directory
   b. Auto-OCR if text empty
   c. Predict (same as single prediction)
   d. Collect results and errors
4. Returns JSON with:
   - total: number of rows
   - successful: number of successful predictions
   - failed: number of failures
   - results: array of predictions
   - errors: array of error messages
5. Results can be exported or reviewed in history
```

### **3. Explainability Generation Workflow**

```
When explain=true in predict request:
1. Generate Grad-CAM heatmap
   - Pass image through model
   - Hook into final conv layer
   - Compute gradient w.r.t. predicted class
   - Create overlay on original image
   
2. Generate LIME explanation
   - Create perturbed versions of image
   - Get predictions for each perturbation
   - Learn linear model explaining decision
   - Highlight important regions
   
3. Extract feature importance
   - Token-level importance for text
   - Spatial importance for vision
   - Attention maps from fusion layer
   
4. Encode visualizations to base64
5. Return all explanations in JSON
```

---

## API Endpoints

### **Core Prediction Endpoints**

#### **POST /api/predict**
Single image prediction with optional explanations.

**Request (multipart/form-data):**
```json
{
  "image": <file>,
  "text": "optional text caption",
  "explain": "false",           // "true" or "false"
  "ocr": "auto"                 // "auto", "true", "false"
}
```

**Response (200):**
```json
{
  "result_id": "uuid-string",
  "timestamp": "2025-02-06T10:30:00.000000",
  "image": "filename.jpg",
  "text": "meme caption text",
  "input_text": "user provided text (if any)",
  "ocr_text": "text extracted by OCR",
  "ocr_engine": "easyocr or tesseract",
  "prediction": {
    "predicted_class": 1,       // 0 = non-hateful, 1 = hateful
    "confidence": 0.85,
    "class_name": "Hateful",
    "probabilities": {
      "0": 0.15,
      "1": 0.85
    }
  },
  "explanations": {             // if explain=true
    "gradcam_base64": "data:image/...",
    "lime_base64": "data:image/...",
    "feature_importance": {...},
    "attention_regions": [...]
  }
}
```

**Error Response (400/500):**
```json
{
  "error": "Description of error"
}
```

---

#### **POST /api/batch-predict**
Batch process multiple images via CSV.

**Request (multipart/form-data):**
```json
{
  "file": <csv_file>
}
```

**CSV Format:**
```
image,text
img001.png,some caption text
img002.png,another meme caption
```

**Response (200):**
```json
{
  "total": 100,
  "successful": 98,
  "failed": 2,
  "results": [
    {
      "row": 1,
      "text_used": "caption",
      "ocr_text": "",
      "ocr_engine": "",
      "predicted_class": 1,
      "confidence": 0.92,
      "class_name": "Hateful"
    }
  ],
  "errors": [
    "Row 5: Image not found - img/missing.png",
    "Row 42: Processing error - invalid image format"
  ]
}
```

---

#### **POST /api/explain**
Detailed explanation for a specific prediction.

**Request:**
```json
{
  "image_path": "img/filename.png",
  "text": "meme text",
  "predicted_class": 1
}
```

**Response:**
```json
{
  "gradcam_base64": "data:image/...",
  "lime_base64": "data:image/...",
  "feature_importance": {
    "text_tokens": {
      "word1": 0.34,
      "word2": 0.28,
      "word3": -0.15
    },
    "attention_weights": [...]
  },
  "attention_regions": [
    {"region": "top-left", "importance": 0.45},
    {"region": "center", "importance": 0.32}
  ]
}
```

---

### **History & Analytics Endpoints**

#### **GET /api/history**
Retrieve prediction history.

**Query Parameters:**
- `limit` (optional): Number of recent predictions to return (default: 50)

**Response:**
```json
[
  {
    "result_id": "...",
    "timestamp": "...",
    "image": "...",
    "text": "...",
    "prediction": {...}
  }
]
```

---

#### **POST /api/clear-history**
Clear all stored prediction history.

**Response:**
```json
{
  "message": "History cleared"
}
```

---

#### **GET /api/stats**
Get aggregate statistics.

**Response:**
```json
{
  "total_predictions": 150,
  "hateful_count": 45,
  "non_hateful_count": 105,
  "hateful_percentage": 30.0,
  "avg_confidence": 0.876
}
```

---

#### **GET /api/health**
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "Hateful Memes Detector v1.0",
  "accuracy": "70.22%",
  "features": [
    "prediction",
    "gradcam",
    "lime",
    "batch_processing",
    "history"
  ]
}
```

---

### **Page Routes**

#### **GET /**
Serves the main prediction interface (index.html)

#### **GET /results**
Displays detailed results for a prediction (results.html)
- Requires `latest_result_id` in session
- Retrieves result from `temp_results/{result_id}.json`

---

## Components Explanation

### **1. app.py - Flask Web Application**

**Key Responsibilities:**
- Initialize Flask app and routes
- Handle file uploads and validation
- Manage prediction workflows
- Handle session management
- Serve HTML templates
- Implement REST API endpoints

**Key Functions:**
- `extract_text_with_ocr()` - Extract text from images using OCR engines
- `should_run_ocr()` - Determine when to run OCR
- `_get_or_create_engine()` - Cache OCR engines for performance
- `index()` - Serve home page
- `predict()` - Main prediction endpoint
- `batch_predict()` - Batch processing
- `explain()` - Generate explanations
- `history()` - Get prediction history
- `stats()` - Get analytics

**Configuration:**
```python
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
TEMP_RESULTS = 'temp_results'
```

---

### **2. predict.py - MemeClassifier**

**Class: MemeClassifier**

**Initialization:**
```python
classifier = MemeClassifier("saved_models/model_best.pt")
```

**Key Methods:**

#### `__init__(model_path, device=None)`
- Loads pre-trained model
- Initializes tokenizer (BERT-like)
- Sets up device (CUDA/CPU auto-detection)
- Loads config from TrainConfig

#### `predict_single(image_path, text, return_confidence=True)`
- Loads and preprocesses image (224×224, RGB)
- Tokenizes text (max 128 tokens)
- Runs inference through model
- Returns prediction with confidence scores

**Key Attributes:**
- `device` - torch.device (cuda or cpu)
- `model` - DualEncoderFusion instance
- `tokenizer` - Text tokenizer
- `cfg` - Training configuration
- `class_names` - {0: "Non-Hateful", 1: "Hateful"}

---

### **3. explainability.py - ExplainabilityEngine**

**Class: ExplainabilityEngine**

**Key Methods:**

#### `explain_prediction(image_path, text, predicted_class)`
Main method that generates all explanations.

**Returns Dictionary with:**
- `gradcam_base64` - Grad-CAM heatmap as base64 image
- `lime_base64` - LIME explanation as base64 image
- `feature_importance` - Text token importance scores
- `attention_regions` - Spatial importance regions
- `confidence_analysis` - Confidence analysis details

#### `_load_group_config()`
- Loads protected group lexicons
- Falls back to defaults if config missing
- Supports categories: religion, ethnicity, nationality, race, orientation, gender, disability, caste

**Key Features:**
- Protected group mention detection
- Threat keyword detection
- Stereotype phrase detection
- Intersectionality analysis (multiple groups)
- Bias metrics reporting

---

### **4. src/training/train_improved.py - DualEncoderFusion Model**

**Model Architecture:**
```
DualEncoderFusion
├── Vision Encoder (Vision Transformer - ViT)
│   └── Outputs: image_features (768 or 1024-dim)
├── Text Encoder (BERT/RoBERTa)
│   └── Outputs: text_features (768 or 1024-dim)
├── Fusion Layer
│   ├── Multimodal Fusion
│   ├── Cross-modal Attention
│   └── Outputs: fused_features
└── Classification Head
    ├── Dense layers
    └── Output: [non-hateful_prob, hateful_prob]
```

**Key Components:**
- **Vision Encoder**: Pre-trained ViT (Vision Transformer)
- **Text Encoder**: Pre-trained BERT/RoBERTa
- **Fusion Mechanism**: Concatenation + attention-based fusion
- **Classification Head**: 2-layer MLP with dropout

**Training Configuration (TrainConfig):**
```python
learning_rate = 2e-5
batch_size = 32
epochs = 10
loss_function = CrossEntropyLoss + Focal Loss
optimizer = AdamW with warmup
```

---

### **5. src/preprocessing/ocr_and_annotations.py - OCR Engine**

**Class: OCREngine**

**Supported Engines:**
1. **EasyOCR** (Primary)
   - GPU-accelerated
   - Multiple language support
   - Faster inference
   
2. **Tesseract** (Fallback)
   - Lightweight
   - CPU-only
   - Reliable baseline

**Key Methods:**
- `read(image_path)` - Extract text from image
- `read_batch(image_paths)` - Batch processing

**Error Handling:**
- Automatic fallback from EasyOCR to Tesseract
- Logs warnings on failure
- Returns empty string if both fail

---

## Model Details

### **Model Specifications**

| Component | Details |
|-----------|---------|
| **Vision Backbone** | Vision Transformer (ViT) |
| **Text Backbone** | BERT/RoBERTa |
| **Fusion Strategy** | Attention-based multimodal fusion |
| **Input Image Size** | 224×224 pixels |
| **Input Text Length** | Max 128 tokens |
| **Output Dimensions** | 2 classes (non-hateful, hateful) |
| **Model Size** | ~340MB |
| **Inference Time** | ~500ms per image (GPU) / ~2s (CPU) |

### **Training Details**

| Parameter | Value |
|-----------|-------|
| **Dataset** | Facebook Hateful Memes Challenge |
| **Train/Val/Test Split** | 8500 / 500 / 500 |
| **Learning Rate** | 2e-5 (AdamW with warmup) |
| **Batch Size** | 32 |
| **Epochs** | 10 |
| **Loss Function** | CrossEntropy + Focal Loss |
| **Validation Metric** | AUROC (70.22%) |
| **Accuracy** | 70.22% |

### **Model Files**

| File | Purpose |
|------|---------|
| `model_best.pt` | Best checkpoint (highest validation accuracy) |
| `model_final.pt` | Final epoch checkpoint |
| `model.pt` | Latest working checkpoint |

---

## Explainability & XAI Features

### **1. Grad-CAM (Gradient-weighted Class Activation Mapping)**

**What it does:**
Highlights which regions of the image were most important for the model's decision.

**How it works:**
1. Forward pass through model
2. Compute gradients w.r.t. predicted class
3. Weight feature maps by gradients
4. Create heatmap overlay
5. Superimpose on original image

**Interpretation:**
- Red/Hot areas: Important for positive class (hateful)
- Blue/Cool areas: Important for negative class (non-hateful)
- Gray areas: Irrelevant regions

**Output:** `gradcam_base64` - Base64 encoded visualization

---

### **2. LIME (Local Interpretable Model-agnostic Explanations)**

**What it does:**
Explains prediction by showing which image regions contribute to the decision.

**How it works:**
1. Create perturbed versions of image (hide regions)
2. Get model predictions for each perturbation
3. Fit linear model to explain differences
4. Extract feature importance from linear model
5. Visualize important regions

**Interpretation:**
- Green regions: Support predicted class
- Red regions: Contradict predicted class
- Opacity: Strength of influence

**Output:** `lime_base64` - Base64 encoded visualization

---

### **3. Feature Importance Analysis**

**Text-level Analysis:**
- **Token Importance**: Integrated Gradients for each word
- **Attention Weights**: Cross-modal attention patterns
- **Semantic Importance**: Which concepts triggered detection

**Vision-level Analysis:**
- **Spatial Importance**: Which image regions were focused
- **Feature Maps**: Visualize learned representations
- **Attention Maps**: Multi-head attention analysis

**Output:** `feature_importance` dictionary with token scores

---

### **4. Protected Group & Bias Detection**

The explainability engine includes built-in detection for:

**Categories:**
- Religion (Muslim, Christian, Hindu, etc.)
- Ethnicity (Arab, Asian, African, etc.)
- Nationality (Indian, Chinese, American, etc.)
- Race (White, Black, Brown, etc.)
- Sexual Orientation (LGBT, Gay, Lesbian, etc.)
- Gender (Male, Female, etc.)
- Immigration Status (Immigrant, Refugee, etc.)
- Disability (Deaf, Blind, Autistic, etc.)
- Caste (Dalit, Brahmin, etc.)

**Threat Detection:**
Identifies threatening language (kill, bomb, burn, etc.)

**Stereotype Detection:**
Identifies stereotype phrases (are terrorists, bring crime, etc.)

**Intersectionality Analysis:**
Identifies memes targeting multiple groups simultaneously

---

## Directory Structure

```
. (root)
│
├── app.py                        # Main Flask web application
├── predict.py                    # Inference module with MemeClassifier
├── explainability.py             # XAI engine (Grad-CAM, LIME, etc.)
│
├── requirements.txt              # Python dependencies
├── requirements-web.txt          # Web-specific dependencies
│
├── config/
│   └── protected_groups.json     # Protected group & stereotype lexicons
│
├── data/
│   ├── raw_memes/               # Input raw meme images
│   ├── annotations.csv          # Image-text-label dataset
│   ├── text/                    # OCR outputs per image
│   └── config/                  # Data configuration files
│
├── img/                          # Actual meme images dataset
│
├── src/                          # Source code modules
│   ├── preprocessing/           # OCR & text extraction
│   │   └── ocr_and_annotations.py
│   ├── training/                # Model training code
│   │   ├── train.py
│   │   ├── train_improved.py    # DualEncoderFusion implementation
│   │   └── train_lite.py
│   ├── inference/               # Inference utilities
│   │   └── predict.py
│   ├── explainability/          # XAI implementations
│   │   └── xai.py
│   ├── evaluation/              # Metrics & evaluation
│   └── ui/                      # Streamlit app (alternative UI)
│
├── static/                       # Frontend assets
│   ├── app.js                   # JavaScript logic
│   └── style.css                # CSS styling
│
├── templates/                    # Flask HTML templates
│   ├── index.html               # Home/prediction page
│   └── results.html             # Results display page
│
├── saved_models/                # Trained model checkpoints
│   ├── model_best.pt            # Best model (recommended)
│   ├── model_final.pt
│   ├── model.pt
│   └── history.json
│
├── reports/                      # Generated reports
│   ├── metrics.json             # Performance metrics
│   └── evaluation_report.json
│
├── results/                      # Prediction results storage
│
├── temp_results/                # Temporary result files (UUID-based)
│   └── {uuid}.json
│
├── uploads/                      # User-uploaded images
│
├── __pycache__/                 # Python cache files
│
├── README.md                    # Original dataset README
├── README_WEB_APP.md            # Web app documentation
├── PROJECT_GUIDE.md             # Project setup guide
├── QUICKSTART.md                # Quick start instructions
├── UI_ENHANCEMENTS.md           # UI improvements documentation
└── LICENSE.txt                  # License information
```

---

## Advanced Features

### **1. Session Management**
- Uses Flask sessions with secure secret key
- Session data stored in signed cookies
- Only stores minimal data (result_id) to avoid 4KB cookie limit
- Actual results stored in JSON files in `temp_results/`

### **2. OCR Engine Caching**
- Initializes OCR engines once and reuses
- Reduces startup time for subsequent requests
- Supports fallback from EasyOCR to Tesseract

### **3. Batch Processing**
- Processes CSV files with multiple memes
- Returns detailed error messages per row
- Tracks successful vs. failed predictions
- Can handle large batches (memory permitting)

### **4. Result Persistence**
- Stores results in JSON files (UUID-based naming)
- Each result is independently retrievable
- Can be extended to database (MongoDB, PostgreSQL, etc.)

### **5. Error Handling**
- Comprehensive try-catch blocks
- Detailed logging with datetime stamps
- Specific HTTP status codes (400, 404, 500)
- User-friendly error messages

### **6. CORS Support**
- Enabled for all routes
- Allows frontend to be hosted separately
- Supports cross-origin API requests

### **7. File Upload Security**
- Max file size: 50MB
- File validation before processing
- Uploaded files stored in isolated directory
- Automatic cleanup possible (not implemented)

---

## Running the Application

### **Development Mode**
```bash
python app.py
# Server starts at http://localhost:5000
```

### **Production Mode**
```bash
# Using gunicorn (install with: pip install gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or using waitress (install with: pip install waitress)
waitress-serve --port=5000 app:app
```

### **With GPU Acceleration**
Automatically detected and used. Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Performance Metrics

### **Inference Performance**
| Hardware | Image Only | With OCR | With Explanations |
|----------|-----------|----------|------------------|
| **GPU (RTX 3080)** | 200ms | 800ms | 2-3s |
| **CPU (Intel i7)** | 2s | 3-4s | 8-10s |

### **Memory Usage**
| Component | Size |
|-----------|------|
| Model (loaded) | ~340MB |
| Python/Libraries | ~2GB |
| EasyOCR Engine | ~500MB |
| Total (startup) | ~3GB |

### **Model Accuracy**
- **AUROC**: 70.22%
- **Accuracy**: 70.22%
- **Precision**: 68.5%
- **Recall**: 72.1%

---

## Troubleshooting

### **Issue: "CUDA out of memory"**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python app.py
```

### **Issue: "OCR engines not initialized"**
```bash
# Install OCR dependencies
pip install easyocr pytesseract
# Also install Tesseract binary from: https://github.com/UB-Mannheim/tesseract/wiki
```

### **Issue: "Model not found"**
```bash
# Ensure model exists at:
# c:\Users\nangi\Downloads\archive (5)\data\saved_models\model_best.pt
```

### **Issue: "Port 5000 already in use"**
```bash
# Use different port
python app.py  # Modify app.run(port=5001) in app.py
```

---

## Integration Examples

### **Python Client Example**
```python
import requests
from pathlib import Path

# Single prediction
with open('meme.jpg', 'rb') as f:
    files = {'image': f}
    data = {'text': 'meme caption', 'explain': 'true'}
    response = requests.post('http://localhost:5000/api/predict', 
                            files=files, data=data)
    result = response.json()
    print(f"Prediction: {result['prediction']['class_name']}")
    print(f"Confidence: {result['prediction']['confidence']}")
```

### **JavaScript Client Example**
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('text', textCaption);
formData.append('explain', 'true');

const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`Prediction: ${result.prediction.class_name}`);
console.log(`Grad-CAM: ${result.gradcam_image}`);
```

---

## License & Attribution

The Hateful Memes Challenge dataset is licensed under the terms in `LICENSE.txt`.

**Dataset Citation:**
```
@inproceedings{kiela2021hateful,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Kiela, Douwe and Firooz, Hamed and Mohan, Amanpreet and Goswami, Vedanuj and Singh, Amanpreet and Ringshia, Pratik and Testuggine, Davide},
  booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
  year={2021}
}
```

---

## Support & Contribution

For issues, enhancements, or contributions, please refer to the project documentation or contact the development team.

**Last Updated:** February 6, 2026

---

## Appendix: Configuration Files

### **protected_groups.json**
```json
{
  "religion": ["muslim", "christian", "hindu", ...],
  "ethnicity": ["arab", "asian", "african", ...],
  "nationality": ["indian", "chinese", "american", ...],
  "stereotypes": ["are terrorists", "bring crime", ...],
  "threats": ["kill", "bomb", "burn", ...]
}
```

### **TrainConfig** (from training module)
```python
class TrainConfig:
    text_model = "bert-base-uncased"  # or roberta-base
    vision_model = "vit_base_patch16_224"
    image_root = Path("img")
    annotations = Path("data/annotations.csv")
    learning_rate = 2e-5
    batch_size = 32
    epochs = 10
    max_token_length = 128
    image_size = 224
```

---

*This comprehensive documentation covers all aspects of the Hateful Memes Detector application, from architecture and tech stack to API usage and troubleshooting.*
