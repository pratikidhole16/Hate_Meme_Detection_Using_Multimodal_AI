# Project Guide

## Folder Layout
- data/raw_memes: place or link your PNG memes (input only). You can link existing img/ via `mklink /J data\raw_memes img` on Windows.
- data/text: OCR outputs per image (.txt).
- data/annotations.csv: combined OCR/text/labels.
- src/preprocessing: OCR + annotation utilities.
- src/training: model and training loop.
- src/inference: prediction helpers.
- src/explainability: Grad-CAM and text attributions.
- src/evaluation: metrics scripts.
- src/ui: Streamlit app.
- saved_models: checkpoints.
- reports: metrics, heatmaps, exports.

## Quickstart
1) Install deps: `pip install -r requirements.txt`
2) Create annotations: `python src/preprocessing/ocr_and_annotations.py --jsonl train.jsonl dev.jsonl --image-root img`
3) Train: `python src/training/train.py --annotations data/annotations.csv --image-root img --output saved_models/model.pt`
4) Evaluate: `python src/evaluation/evaluate.py --checkpoint saved_models/model.pt --annotations data/annotations.csv --image-root img`
5) UI: `streamlit run src/ui/app.py`

## Labeling Guidance
- Columns: image_id, text, label (0=non-hate, 1=implicit, 2=explicit). The provided script fills label=-1 when unknown so you can annotate manually.
- For active learning: sample low-confidence items using evaluation probabilities and relabel.

## XAI
- Vision: Grad-CAM heatmaps saved to reports/heatmaps.
- Text: Integrated Gradients token importances saved alongside heatmaps.
