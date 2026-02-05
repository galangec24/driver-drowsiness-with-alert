# data_diagnostic.py
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset(dataset_path):
    """Analyze the dataset for potential issues"""
    
    classes = ["Drowsy", "Non_Drowsy", "Yawning"]
    
    print("🔍 ANALYZING DATASET...")
    print("="*50)
    
    all_features = []
    all_labels = []
    
    for i, cls in enumerate(classes):
        cls_path = os.path.join(dataset_path, cls)
        if not os.path.exists(cls_path):
            print(f"❌ {cls} folder not found!")
            continue
            
        images = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))
        print(f"\n📊 {cls}: {len(images)} images")
        
        # Analyze sample images
        sample_images = images[:10]
        brightness_values = []
        contrast_values = []
        
        for img_path in sample_images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  ⚠️ Corrupted: {os.path.basename(img_path)}")
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            brightness_values.append(brightness)
            contrast_values.append(contrast)
            
            # Extract simple features (for analysis only)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten()
            all_features.append(hist)
            all_labels.append(i)
        
        if brightness_values:
            print(f"  • Avg brightness: {np.mean(brightness_values):.1f}")
            print(f"  • Avg contrast: {np.mean(contrast_values):.1f}")
            print(f"  • Brightness range: {min(brightness_values):.1f}-{max(brightness_values):.1f}")
    
    # Check if classes are separable with simple features
    if len(all_features) > 0:
        print("\n" + "="*50)
        print("🎯 TESTING CLASS SEPARABILITY")
        print("="*50)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a simple classifier
        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Simple SVM accuracy: {acc:.1%}")
        
        if acc > 0.7:
            print("✅ Good: Classes seem separable")
        elif acc > 0.5:
            print("⚠️ Fair: Some class confusion")
        else:
            print("❌ Poor: Classes may be poorly defined or mixed up")
    
    # Visualize sample images
    print("\n" + "="*50)
    print("👀 VISUALIZING SAMPLE IMAGES")
    print("="*50)
    
    plt.figure(figsize=(15, 10))
    
    for i, cls in enumerate(classes):
        cls_path = os.path.join(dataset_path, cls)
        images = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.png"))
        
        if not images:
            continue
            
        # Show 3 sample images per class
        for j in range(min(3, len(images))):
            img = cv2.imread(images[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 3, i*3 + j + 1)
            plt.imshow(img)
            plt.title(f"{cls} - Sample {j+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=100)
    print("✅ Sample images saved to dataset_samples.png")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_path = os.path.join(project_root, "datasets", "complete_drowsiness")
    
    if os.path.exists(dataset_path):
        analyze_dataset(dataset_path)
    else:
        print(f"❌ Dataset not found at {dataset_path}")