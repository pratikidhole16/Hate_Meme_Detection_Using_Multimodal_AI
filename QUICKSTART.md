# 🚀 Quick Start Guide - Hateful Memes Detector

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (4GB+ VRAM recommended) OR CPU mode
- 2GB disk space for models

## 🎯 Installation (5 minutes)

### Step 1: Install Dependencies
```bash
# Navigate to project directory
cd "c:\Users\nangi\Downloads\archive (5)\data"

# Install all requirements
pip install -r requirements.txt
pip install -r requirements-web.txt
pip install grad-cam  # For visual explanations
```

### Step 2: Verify Model Files
```bash
# Check if trained model exists
ls saved_models/
# Should see: best_model.pt (or similar)
```

### Step 3: Start the Web App
```bash
# Launch Flask server
python src/ui/app.py
```

### Step 4: Access the Application
Open your browser and go to:
```
http://localhost:5000
```

## 📱 How to Use

### 1. **Predict Tab** (Main Feature)
   - **Upload Image**: Click upload area or drag-drop a meme image
   - **Add Caption**: Enter text from the meme (optional)
   - **Enable XAI**: Check box to get explanations (Grad-CAM, LIME, etc.)
   - **Click Analyze**: Get instant prediction with confidence

### 2. **Understand Results**
   - **Green ✅**: Safe content (not hateful)
   - **Red 🚨**: Hateful content detected
   - **Confidence %**: How certain the model is (0-100%)
   - **Heatmap**: Visual attention map showing influential regions
   - **Keywords**: Highlighted text showing importance

### 3. **History Tab**
   - View all previous predictions
   - Timestamp for each analysis
   - Click to see full details

### 4. **Statistics Tab**
   - Overall accuracy metrics
   - Prediction distribution
   - Performance charts

## 🎨 Modern UI Features

### Animations
- Smooth fade-in/out transitions
- Animated confidence bar
- Staggered keyword highlighting
- Result container persistence (doesn't disappear!)

### Design
- Dark theme with purple/pink accents
- Glassmorphism effect (frosted glass style)
- Responsive design (works on mobile, tablet, desktop)
- Interactive hover effects

### Accessibility
- Color-coded predictions (green/red)
- High-contrast text
- Keyboard navigation support
- Screen reader friendly

## 🔧 Configuration

### Model Settings
In `app.py`, adjust these variables:
```python
CHECKPOINT = Path("saved_models/best_model.pt")
TEXT_MODEL = "distilbert-base-uncased"
VISION_MODEL = "convnext_base"
MAX_CONFIDENCE_THRESHOLD = 0.95
```

### Flask Settings
```python
DEBUG = False  # Set to True for development
UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
```

## 📊 API Endpoints

### Prediction
```bash
POST /api/predict
Content-Type: multipart/form-data

Parameters:
- image: Image file
- text: Meme caption text
- explain: Boolean (true for explanations)

Response:
{
  "prediction": {
    "class": 0,
    "class_name": "Safe Content",
    "confidence": 0.85,
    "class_probabilities": {
      "Safe": 0.85,
      "Hateful": 0.15
    }
  },
  "explanations": {
    "reasoning": "...",
    "gradcam_image": "...",
    "text_importance": {...},
    "attention_regions": [...]
  }
}
```

### History
```bash
GET /api/history

Response: [
  {
    "id": 1,
    "timestamp": "2025-01-15 14:30:22",
    "prediction": "Safe Content",
    "confidence": 0.85,
    "image_path": "..."
  },
  ...
]
```

### Statistics
```bash
GET /api/stats

Response: {
  "total_predictions": 42,
  "safe_count": 32,
  "hateful_count": 10,
  "accuracy": 0.7022,
  "avg_confidence": 0.78
}
```

## 🐛 Troubleshooting

### Issue: "Port 5000 already in use"
```bash
# Kill existing Flask process
netstat -ano | find "5000"
taskkill /PID <PID> /F

# Or use different port
set FLASK_PORT=5001
python src/ui/app.py
```

### Issue: "Model not found"
```bash
# Verify model path
dir saved_models\
# Make sure best_model.pt exists
```

### Issue: "CUDA out of memory"
```bash
# Use CPU mode
set CUDA_VISIBLE_DEVICES=
python src/ui/app.py
```

### Issue: "Results disappearing" ✅ FIXED
- This was a bug in JavaScript that cleared the results container
- Now fixed with proper DOM persistence
- Results will stay visible until you upload a new image or click "Clear Image"

## 📈 Performance Tips

### Faster Inference
- Use GPU: Ensure CUDA is available (`torch.cuda.is_available()`)
- Reduce image size: Resize to 512x512 before upload
- Disable explanations: Uncheck XAI for faster prediction

### Better Accuracy
- Ensure good image quality
- Include full caption text
- Use diverse images during training

## 🔐 Security

### File Upload Safety
- Max 16MB file size
- Only JPG, PNG, GIF allowed
- Files saved temporarily, auto-deleted

### Model Safety
- No personally identifiable information (PII) stored
- Predictions not logged permanently by default
- HTTPS recommended for production

## 📚 Documentation

- **`README.md`**: Project overview
- **`PROJECT_GUIDE.md`**: Data and methodology
- **`UI_ENHANCEMENTS.md`**: Modern design details
- **`src/training/train_improved.py`**: Model training code
- **`src/inference/predict.py`**: Inference pipeline

## 🎓 Educational Use

### Understanding the Model
1. Read `PROJECT_GUIDE.md` for methodology
2. Check `src/training/train_improved.py` for architecture
3. Review evaluation metrics in `evaluate.py`

### Modifying the Model
```python
# In train_improved.py
class DualEncoderFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Modify architecture here
        self.vision = create_vision_encoder()  # Customizable
        self.text = create_text_encoder()      # Customizable
        # Add layers as needed
```

## 🆘 Support

### Common Questions

**Q: Can I use my own images?**
A: Yes! Upload any image with hateful meme text.

**Q: Is the model always accurate?**
A: No, it's ~70% accurate. False positives/negatives are possible.

**Q: Can I download explanations?**
A: Yes, the API returns all explanation data as JSON.

**Q: Does it save my images?**
A: No, images are temporary and auto-deleted.

**Q: Can I run this offline?**
A: Yes, once downloaded and models loaded, no internet required.

## 📦 Deployment

### Local Deployment (Done! ✅)
```bash
python src/ui/app.py
# Access: http://localhost:5000
```

### Production Deployment (AWS/Azure/GCP)
```bash
# 1. Install gunicorn
pip install gunicorn

# 2. Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.ui.app:app

# 3. Use reverse proxy (Nginx/Apache) for HTTPS
```

### Docker Deployment
```bash
# Build image
docker build -t hateful-memes-detector .

# Run container
docker run -p 5000:5000 hateful-memes-detector
```

## ✅ Checklist

Before using in production:
- [ ] Test with various image types
- [ ] Verify GPU/CPU mode works
- [ ] Check file upload limits
- [ ] Test on mobile device
- [ ] Enable HTTPS
- [ ] Set up monitoring
- [ ] Create backup of models
- [ ] Document custom changes

## 🎉 You're Ready!

The Hateful Memes Detector is now fully operational with:
- ✅ Modern, responsive UI
- ✅ Fast inference engine
- ✅ Explainable AI (XAI)
- ✅ Result persistence (bug fixed!)
- ✅ Beautiful animations
- ✅ Production-ready Flask app

**Start analyzing memes now!** 🚀

---

**Last Updated**: January 2025  
**Version**: 1.0 - Production Ready
