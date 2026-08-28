# 🎨 Hateful Memes Detector - Modern Web Application

A production-ready Flask web application with advanced XAI (Explainable AI) features for detecting hateful content in memes. Built with modern tech stack and beautiful UI.

## ✨ Features

### 🎯 Core Functionality
- **Real-time Prediction**: Analyze memes with image + text input
- **Confidence Scores**: Get detailed probability distributions
- **Batch Processing**: Process multiple memes from CSV files
- **Prediction History**: View all previous predictions
- **Live Statistics**: Dashboard with analytics and trends

### 🧠 Explainability Features (XAI)
- **Grad-CAM Visualization**: See which regions the model focuses on
- **LIME Explanations**: Understand individual feature importance
- **Saliency Maps**: Gradient-based input sensitivity analysis
- **Text Analysis**: Identify important keywords in captions
- **Attention Regions**: Highlights suspicious patterns
- **Detailed Reasoning**: Human-readable explanations for predictions

### 🎨 Modern UI
- **Glassmorphism Design**: Beautiful frosted glass effects
- **Dark Mode**: Eye-friendly dark theme with purple accents
- **Drag & Drop**: Easy image upload experience
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Instant feedback and results
- **Smooth Animations**: Professional transitions and effects

## 🏗️ Tech Stack

**Backend**
- Flask 2.3+ - Web framework
- PyTorch 2.1+ - Deep learning
- Transformers 4.37+ - NLP models
- TIMM - Vision models
- LIME - Local Interpretability
- Grad-CAM - Visual explanations

**Frontend**
- HTML5 - Semantic markup
- Tailwind CSS - Modern styling
- Vanilla JavaScript - Interactivity
- Chart.js - Statistics visualization
- GSAP - Smooth animations

**Infrastructure**
- Python 3.8+
- CUDA 12.1 (optional, for GPU)

## 📦 Installation

### 1. Clone/Setup Project
```bash
cd /path/to/project
```

### 2. Install Dependencies
```bash
pip install -r requirements-web.txt
```

### 3. Verify Model Files
Ensure these exist:
- `saved_models/model_best.pt` - Trained model (70.22% accuracy)
- `predict.py` - Prediction interface
- `explainability.py` - XAI engine

### 4. Create Directories
```bash
mkdir -p uploads results templates static
```

## 🚀 Running the Application

### Development Mode
```bash
python app.py
```

Then open: **http://localhost:5000**

### Production Mode (Gunicorn)
```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

### With Custom Port
```bash
python app.py --port 8080
```

## 📖 Usage Guide

### Single Image Prediction
1. **Upload Image**: Drag & drop or click to upload meme
2. **Add Caption**: Enter the meme text/caption
3. **Enable Explanations**: Check "Include XAI Explanations"
4. **Analyze**: Click "Analyze Meme"

### Understanding Results

| Component | Meaning |
|-----------|---------|
| **Classification** | `Hateful` or `Non-Hateful` |
| **Confidence** | How certain the model is (0-100%) |
| **Grad-CAM** | Heat map of visual focus regions |
| **Text Analysis** | Color-coded word importance |
| **Attention Regions** | Key observations and reasons |
| **Reasoning** | Detailed explanation of decision |

### Batch Processing
```bash
# Create CSV with columns: image, text
# Example: batch_data.csv
# image,text
# img1.png,This is hateful
# img2.png,This is funny

# Then upload via UI or API
curl -F "file=@batch_data.csv" http://localhost:5000/api/batch-predict
```

## 🔌 API Endpoints

### Predict Single Image
```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@meme.png" \
  -F "text=Meme caption" \
  -F "explain=true"
```

### Get Prediction History
```bash
curl http://localhost:5000/api/history?limit=50
```

### Get Statistics
```bash
curl http://localhost:5000/api/stats
```

### Health Check
```bash
curl http://localhost:5000/api/health
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 70.22% |
| **F1 Score** | 0.6680 |
| **Precision** | 0.6762 |
| **Recall** | 0.6640 |
| **Architecture** | ConvNext-Base + DistilBERT |

## 🎓 Understanding XAI Features

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Shows which image regions influenced the prediction
- Red areas = high importance
- Blue areas = low importance
- Helps validate if model focuses on relevant features

### LIME (Local Interpretable Model-agnostic Explanations)
- Tests small variations of the image
- Identifies pixels most important for prediction
- Provides model-agnostic explanations
- Works with any prediction model

### Saliency Maps
- Computes gradient of output w.r.t. input
- Shows input sensitivity to model output
- Higher values = more influential pixels

### Text Analysis
- Analyzes caption/text contribution
- Highlights suspicious keywords
- Shows word importance scores
- Color-coded (red=hateful, green=neutral)

## 🛠️ Configuration

Edit `app.py` to customize:
```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Max file size
app.config['UPLOAD_FOLDER'] = Path('uploads')        # Upload directory
DEBUG = True  # Debug mode
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `app.py`
- Use CPU: Set `device='cpu'` in `predict.py`

### Model Not Found
- Check `saved_models/model_best.pt` exists
- Run training script: `python src/training/train_improved.py`

### Slow Predictions
- First prediction is slower (model loading)
- Subsequent predictions are cached in memory
- Use GPU for 2-5x speedup

## 📈 Future Enhancements

- [ ] Database integration (SQLite/PostgreSQL)
- [ ] User authentication
- [ ] Advanced filtering and search
- [ ] Model retraining pipeline
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Mobile app
- [ ] Real-time notifications
- [ ] A/B testing framework

## 📄 License

MIT License - See LICENSE.txt

## 👨‍💻 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

- Issues: Check GitHub Issues
- Documentation: See inline code comments
- Model Details: Check `saved_models/history.json`

## 🎉 Credits

- Model trained on Hateful Memes Dataset
- XAI methods from PyTorch ecosystem
- UI inspired by modern design systems
- Built with ❤️ using Flask & PyTorch

---

**Status**: ✅ Production Ready | **Accuracy**: 70.22% | **Version**: 1.0.0
