import json
import joblib
import os

models_dir = "../models"

print("="*80)
print("MODEL DIAGNOSTIC")
print("="*80)

# Check driver mapping
with open(os.path.join(models_dir, "driver_mapping.json"), 'r') as f:
    mapping = json.load(f)
print(f"Driver Mapping: {mapping}")

# Check SVM classes
driver_svm = joblib.load(os.path.join(models_dir, "driver_svm.pkl"))
print(f"\nSVM Classes: {driver_svm.classes_}")
print(f"Number of classes: {len(driver_svm.classes_)}")

# Check encoder
driver_encoder = joblib.load(os.path.join(models_dir, "driver_encoder.pkl"))
print(f"\nEncoder classes: {driver_encoder.classes_}")

# Check scaler
driver_scaler = joblib.load(os.path.join(models_dir, "driver_scaler.pkl"))
print(f"\nScaler expects {driver_scaler.mean_.shape[0]} features")