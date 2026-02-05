# test_model_classes.py
import os
import json
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, "models")

# Load drowsiness classes
classes_path = os.path.join(models_dir, "drowsiness_classes.json")
with open(classes_path, 'r') as f:
    drowsiness_info = json.load(f)

print("="*60)
print("CURRENT MODEL CLASSES")
print("="*60)
print(f"Class indices: {drowsiness_info['class_indices']}")
print(f"Classes: {drowsiness_info['classes']}")

# The issue: Model has 'Non_Drowsy' but unified_detector expects 'Alert'