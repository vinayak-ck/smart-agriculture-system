# Datasets & Model Training Guide

## 1. Disease Detection — CNN Model

### Download Dataset
**PlantVillage Dataset** (most widely used, free)
- Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Direct (open): https://github.com/spMohanty/PlantVillage-Dataset
- Rice-specific: https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset

Rice disease classes included:
- Bacterial Leaf Blight, Brown Spot, Leaf Blast, Sheath Blight, Tungro, Healthy

### Train the CNN (ResNet18)
```python
# train_disease_model.py
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score
from pathlib import Path

DATASET_PATH = "path/to/plantvillage"   # folder with subfolders per class
MODEL_OUT    = "ml_models/disease_model.pth"
ACC_OUT      = "ml_models/disease_model_accuracy.txt"
CLASSES      = 8     # match DISEASE_CLASSES in ml_utils.py
EPOCHS       = 15
BATCH        = 32

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=.2, contrast=.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

full = ImageFolder(DATASET_PATH, transform=transform_train)
val_size  = int(0.2 * len(full))
train_set, val_set = torch.utils.data.random_split(full, [len(full)-val_size, val_size])
val_set.dataset.transform = transform_val

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, CLASSES)
model  = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds  = torch.argmax(model(images), dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    acc = round(accuracy_score(all_labels, all_preds) * 100, 1)
    print(f"Epoch {epoch+1}/{EPOCHS} — Val Accuracy: {acc}%")

Path(MODEL_OUT).parent.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)
Path(ACC_OUT).write_text(str(acc))
print(f"Saved model to {MODEL_OUT} | Accuracy: {acc}%")
```

Run:
```bash
python train_disease_model.py
```
Then restart Django — it auto-loads `ml_models/disease_model.pth`.

---

## 2. Growth / Yield Regression Model

The growth model is **auto-trained on startup** using synthetic data
(1000 samples, Gradient Boosting Regressor).

### Improve with Real Data
Once you collect field data (2–3 weeks of sensor readings + actual yield),
retrain with your real CSV:

```python
# retrain_growth_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# Your CSV columns: ph,npk_n,npk_p,npk_k,temperature,humidity,moisture,day,yield_per_acre,height_cm
df = pd.read_csv("field_data.csv").dropna()
features = ['ph','npk_n','npk_p','npk_k','temperature','humidity','moisture','day']
X = df[features].values

for target, key in [('yield_per_acre','yield_model'), ('height_cm','height_model')]:
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    model   = GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=5)
    model.fit(X_tr_s, y_tr)
    acc = round(r2_score(y_te, model.predict(scaler.transform(X_te))) * 100, 1)
    print(f"{key} R²: {acc}%")

# Save (joblib format matches ml_utils.py)
joblib.dump({'yield_model': ..., 'height_model': ..., 'scaler': scaler,
             'yield_accuracy': acc, 'height_accuracy': acc},
            'ml_models/growth_model.pkl')
```

---

## Expected Accuracy (with PlantVillage dataset)

| Model | Type | Expected Accuracy |
|-------|------|-------------------|
| Disease CNN | ResNet18, transfer learning | 92–97% |
| Growth Yield | Gradient Boosting Regressor | 88–93% R² |
| Height | Gradient Boosting Regressor | 97–99% R² |
