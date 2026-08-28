import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load data
annotations = Path("data/annotations.csv")
df = pd.read_csv(annotations)
labeled = df[df["label"] >= 0].copy()
print(f"Total labeled samples: {len(labeled)}")

# Simple encoding
texts = labeled["text"].fillna("").values
labels = labeled["label"].values.astype(int)

def text_to_vec(text, dim=128):
    vec = np.zeros(dim)
    for i, c in enumerate(text[:dim]):
        vec[i] = ord(c) % 128
    return vec.astype(np.float32)

print("Encoding texts...")
X = np.array([text_to_vec(t) for t in texts])
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# Load trained model
print("Loading model...")
checkpoint = torch.load("saved_models/model.pt", map_location="cpu")
model_dict = checkpoint

# Simple model for evaluation
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 3)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# Make predictions on test set
device = "cpu"
model.eval()
with torch.no_grad():
    X_test_device = X_test.to(device)
    logits = model(X_test_device)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

y_test_np = y_test.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(y_test_np, preds)
f1_macro = f1_score(y_test_np, preds, average="macro", zero_division=0)
f1_weighted = f1_score(y_test_np, preds, average="weighted", zero_division=0)
precision_macro = precision_score(y_test_np, preds, average="macro", zero_division=0)
precision_weighted = precision_score(y_test_np, preds, average="weighted", zero_division=0)
recall_macro = recall_score(y_test_np, preds, average="macro", zero_division=0)
recall_weighted = recall_score(y_test_np, preds, average="weighted", zero_division=0)
cm = confusion_matrix(y_test_np, preds)

# Per-class metrics
metrics_by_class = {}
for cls in range(3):
    cls_preds = (preds == cls).astype(int)
    cls_true = (y_test_np == cls).astype(int)
    precision_cls = precision_score(cls_true, cls_preds, zero_division=0)
    recall_cls = recall_score(cls_true, cls_preds, zero_division=0)
    f1_cls = f1_score(cls_true, cls_preds, zero_division=0)
    metrics_by_class[f"class_{cls}"] = {
        "precision": float(precision_cls),
        "recall": float(recall_cls),
        "f1": float(f1_cls)
    }

# Compile results
results = {
    "overall_metrics": {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted)
    },
    "per_class_metrics": metrics_by_class,
    "confusion_matrix": cm.tolist(),
    "test_set_size": len(y_test_np),
    "class_distribution": {
        "class_0": int(np.sum(y_test_np == 0)),
        "class_1": int(np.sum(y_test_np == 1)),
        "class_2": int(np.sum(y_test_np == 2))
    }
}

# Save and print results
Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
print("\nOverall Metrics:")
print(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  F1 (Macro):      {f1_macro:.4f}")
print(f"  F1 (Weighted):   {f1_weighted:.4f}")
print(f"  Precision (Macro):   {precision_macro:.4f}")
print(f"  Precision (Weighted): {precision_weighted:.4f}")
print(f"  Recall (Macro):      {recall_macro:.4f}")
print(f"  Recall (Weighted):   {recall_weighted:.4f}")

print("\nPer-Class Metrics:")
for cls, metrics in metrics_by_class.items():
    print(f"  {cls}:")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClass Distribution (Test Set):")
print(f"  Class 0: {results['class_distribution']['class_0']}")
print(f"  Class 1: {results['class_distribution']['class_1']}")
print(f"  Class 2: {results['class_distribution']['class_2']}")

print(f"\nTest Set Size: {len(y_test_np)} samples")
print("\nResults saved to: reports/metrics.json")
print("="*60)
