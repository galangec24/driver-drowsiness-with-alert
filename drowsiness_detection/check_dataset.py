# check_dataset.py
import os
from glob import glob

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets = [
    os.path.join(project_root, "datasets", "balanced_drowsiness"),
    os.path.join(project_root, "datasets", "complete_drowsiness")
]

for dataset_path in datasets:
    if os.path.exists(dataset_path):
        print(f"\n📁 Dataset: {dataset_path}")
        for class_folder in ["Drowsy", "Non_Drowsy", "Yawning"]:
            class_path = os.path.join(dataset_path, class_folder)
            if os.path.exists(class_path):
                images = glob(os.path.join(class_path, "*.jpg")) + glob(os.path.join(class_path, "*.png"))
                print(f"  {class_folder}: {len(images)} images")