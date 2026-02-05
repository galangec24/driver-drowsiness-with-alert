"""
ENHANCED DRIVER RECOGNITION MODEL TRAINING
Optimized for small dataset (3 images per driver)
"""

import os
import json
import cv2
import numpy as np
import joblib
import sqlite3
from datetime import datetime
from glob import glob
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

print("="*80)
print("👤 ENHANCED DRIVER RECOGNITION TRAINING (Few-Shot Learning)")
print("="*80)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, "models")
face_images_dir = os.path.join(project_root, "backend", "face_images")
database_path = os.path.join(project_root, "backend", "drivers.db")

print(f"📁 Project root: {project_root}")
print(f"📁 Models directory: {models_dir}")
print(f"📁 Face images: {face_images_dir}")
print(f"📁 Database: {database_path}")

# Create models directory
os.makedirs(models_dir, exist_ok=True)

# Check if database exists
if not os.path.exists(database_path):
    print(f"❌ Database not found at: {database_path}")
    print("💡 Please run the backend server first to create the database")
    exit()

# Check face images directory
if not os.path.exists(face_images_dir):
    print(f"❌ Face images directory not found!")
    print("💡 Register drivers and capture face images using the PWA first")
    exit()

def get_driver_info_from_db():
    """Get driver information from database with validation"""
    conn = None
    try:
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all drivers
        cursor.execute('''
            SELECT driver_id, name, phone, email, reference_number, license_number, registration_date
            FROM drivers
            ORDER BY registration_date DESC
        ''')
        
        drivers = {}
        for row in cursor.fetchall():
            driver_id = row['driver_id']
            driver_name = row['name']
            
            # Check folder
            driver_folder = os.path.join(face_images_dir, driver_id)
            folder_exists = os.path.exists(driver_folder)
            
            # Count images in folder
            folder_image_count = 0
            if folder_exists:
                images = glob(os.path.join(driver_folder, "*.jpg")) + glob(os.path.join(driver_folder, "*.png"))
                
                # Verify images are valid
                valid_images = []
                for img_path in images:
                    try:
                        img = cv2.imread(img_path)
                        if img is not None and img.size > 0:
                            valid_images.append(img_path)
                    except:
                        pass
                
                folder_image_count = len(valid_images)
            
            drivers[driver_id] = {
                'name': driver_name,
                'phone': row['phone'],
                'email': row['email'],
                'reference_number': row['reference_number'],
                'license_number': row['license_number'],
                'registration_date': row['registration_date'],
                'folder_exists': folder_exists,
                'folder_image_count': folder_image_count,
                'has_sufficient_images': folder_image_count >= 3
            }
        
        conn.close()
        return drivers
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            conn.close()
        return {}
    except Exception as e:
        print(f"❌ Error getting driver info: {e}")
        return {}

# Get driver info
print("\n📊 Fetching driver information...")
drivers_info = get_driver_info_from_db()

if not drivers_info:
    print("❌ No drivers found in database!")
    print("💡 Register drivers using the PWA first")
    exit()

print(f"\n📋 Found {len(drivers_info)} drivers in database:")
for driver_id, info in drivers_info.items():
    status = "✅" if info['has_sufficient_images'] else "❌"
    print(f"  {status} {driver_id} - {info['name']}")
    print(f"     • Images: {info['folder_image_count']}/3")

# Filter drivers with sufficient images
valid_drivers = {driver_id: info for driver_id, info in drivers_info.items() 
                if info['has_sufficient_images']}

if len(valid_drivers) < 2:  # Need at least 2 drivers
    print("\n❌ Need at least 2 drivers with sufficient face images!")
    print("💡 Minimum: 3 images per driver")
    exit()

print(f"\n🎯 {len(valid_drivers)} drivers have sufficient images for training:")

# Create driver mapping
driver_names = {}
driver_images_count = {}

for driver_id, info in valid_drivers.items():
    driver_names[driver_id] = info['name']
    folder_path = os.path.join(face_images_dir, driver_id)
    images = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
    
    # Count valid images
    valid_count = 0
    for img_path in images:
        try:
            img = cv2.imread(img_path)
            if img is not None and img.size > 0:
                valid_count += 1
        except:
            pass
    
    driver_images_count[driver_id] = min(valid_count, 3)  # Use max 3 images per driver
    print(f"  • {driver_id} ({info['name']}): {driver_images_count[driver_id]} valid images")

# Initialize MediaPipe for enhanced feature extraction
print("\n🔍 Initializing face analysis...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_robust_features(image_path):
    """Extract robust facial features for few-shot learning"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        features = []
        
        # 1. Basic image statistics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Brightness and contrast
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 128.0
        features.extend([brightness, contrast])
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist.tolist())
        
        # 2. Face mesh landmarks (most important)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Use key landmarks only (to avoid too many features)
            key_indices = [
                33, 133,  # Left eye corners
                362, 263, # Right eye corners
                1,  # Nose tip
                61, 291,  # Mouth corners
                10, 152,  # Chin and forehead
                78, 308,  # Cheeks
                54, 284,  # Eyebrows
            ]
            
            for idx in key_indices:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    features.extend([lm.x, lm.y])
                else:
                    features.extend([0.5, 0.5])
                    
            # Add face width/height from landmarks
            if len(landmarks.landmark) > 0:
                xs = [lm.x for lm in landmarks.landmark]
                ys = [lm.y for lm in landmarks.landmark]
                face_width = (max(xs) - min(xs)) * w
                face_height = (max(ys) - min(ys)) * h
                features.extend([face_width / w, face_height / h])
            else:
                features.extend([0.3, 0.4])
        else:
            # No face detected, use placeholder
            features.extend([0.5, 0.5] * len(key_indices) * 2)
            features.extend([0.3, 0.4])
        
        # 3. LBP-like texture features (simplified)
        lbp_blocks = []
        if h > 40 and w > 40:
            for y in range(0, h, h//3)[:3]:
                for x in range(0, w, w//3)[:3]:
                    block = gray[y:y+h//3, x:x+w//3]
                    if block.size > 0:
                        block_mean = np.mean(block) / 255.0
                        block_std = np.std(block) / 128.0
                        lbp_blocks.extend([block_mean, block_std])
                    else:
                        lbp_blocks.extend([0.5, 0.5])
        else:
            lbp_blocks = [0.5, 0.5] * 9
        
        features.extend(lbp_blocks[:9])  # Use first 9 texture features
        
        # 4. Color moments (skin tone)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_mean = np.mean(hsv[:,:,0]) / 180.0
        sat_mean = np.mean(hsv[:,:,1]) / 255.0
        features.extend([hue_mean, sat_mean])
        
        # Ensure fixed feature count (50 features for few-shot learning)
        target_features = 50
        if len(features) > target_features:
            features = features[:target_features]
        elif len(features) < target_features:
            features.extend([0.0] * (target_features - len(features)))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        print(f"    ⚠️ Error processing {os.path.basename(image_path)}: {str(e)[:50]}")
        return None

# Collect training data (limited to 3 images per driver)
print("\n📊 Collecting training data (max 3 images per driver)...")
X = []
y = []
failed_images = []
processed_counts = {}
samples_per_driver = 3  # Fixed to handle limited data

# Process each driver
for driver_id, info in valid_drivers.items():
    folder_path = os.path.join(face_images_dir, driver_id)
    images = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
    
    # Take only first N images
    images = images[:samples_per_driver]
    
    print(f"\n📸 Processing {info['name']} ({driver_id}):")
    
    valid_count = 0
    for img_path in images:
        features = extract_robust_features(img_path)
        
        if features is not None:
            X.append(features)
            y.append(driver_id)
            valid_count += 1
        else:
            failed_images.append(os.path.basename(img_path))
    
    processed_counts[driver_id] = valid_count
    print(f"  ✅ {valid_count}/{len(images)} images processed successfully")

# Check if we have enough data
if len(X) < len(valid_drivers) * 2:  # Need at least 2 samples per driver
    print(f"\n❌ Insufficient training data!")
    print(f"   Need at least 2 samples per driver, have {len(X)} total")
    exit()

X = np.array(X)
y = np.array(y)

print(f"\n📈 Dataset summary:")
print(f"  • Total samples: {X.shape[0]}")
print(f"  • Features per sample: {X.shape[1]}")
print(f"  • Unique drivers: {len(np.unique(y))}")
print(f"  • Failed images: {len(failed_images)}")

# Show samples per driver
print(f"\n📊 Samples per driver:")
for driver_id, count in processed_counts.items():
    print(f"  • {driver_names[driver_id]}: {count} samples")

# FEW-SHOT LEARNING APPROACH
print("\n🎯 Using Few-Shot Learning Approach (Leave-One-Out Cross Validation)")
print("   This is optimal for small datasets (3 images per driver)")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\n🔤 Label encoding:")
for i, driver_id in enumerate(encoder.classes_):
    print(f"  • Class {i}: {driver_names[driver_id]} ({driver_id})")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use models optimized for few-shot learning
print("\n🤖 Training with few-shot optimized models...")

models = {
    'SVM_RBF': SVC(kernel='rbf', C=100, gamma='auto', probability=True, random_state=42),
    'SVM_Linear': SVC(kernel='linear', C=10, probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(
        n_estimators=50, 
        max_depth=5,  # Shallow trees for small data
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
}

# Use Leave-One-Out Cross Validation (optimal for small datasets)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

best_model = None
best_accuracy = 0
best_model_name = ""
cv_scores = {}

print("\n📊 Cross-Validation Results (Leave-One-Out):")
for name, model in models.items():
    print(f"\n  Evaluating {name}...")
    
    # Cross-validation
    scores = cross_val_score(model, X_scaled, y_encoded, cv=loo, scoring='accuracy')
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    cv_scores[name] = {
        'mean': mean_score,
        'std': std_score,
        'scores': scores.tolist()
    }
    
    print(f"    Mean CV Accuracy: {mean_score:.1%} (±{std_score:.1%})")
    
    if mean_score > best_accuracy:
        best_accuracy = mean_score
        best_model = model
        best_model_name = name

# Train the best model on all data
print(f"\n🏆 Best model: {best_model_name} with {best_accuracy:.1%} CV accuracy")
best_model.fit(X_scaled, y_encoded)

# Generate detailed performance analysis
print("\n📊 Detailed Performance Analysis:")

# Confusion matrix using leave-one-out predictions
loo_predictions = []
loo_true = []

for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    # Clone and train model
    model_clone = SVC(kernel='rbf', C=100, gamma='auto') if best_model_name.startswith('SVM') else \
                  RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model_clone.fit(X_train, y_train)
    
    # Predict
    pred = model_clone.predict(X_test)
    loo_predictions.append(pred[0])
    loo_true.append(y_test[0])

# Confusion matrix
cm = confusion_matrix(loo_true, loo_predictions)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[driver_names[cls] for cls in encoder.classes_],
            yticklabels=[driver_names[cls] for cls in encoder.classes_])
plt.title(f'Driver Recognition Confusion Matrix\n(Leave-One-Out CV, Accuracy: {best_accuracy:.1%})')
plt.ylabel('True Driver')
plt.xlabel('Predicted Driver')
plt.tight_layout()
cm_path = os.path.join(models_dir, "driver_recognition_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
print(f"✅ Confusion matrix saved to: {cm_path}")

# Class-wise accuracy
print("\n📊 Class-wise Accuracy:")
class_accuracies = {}
for i, driver_id in enumerate(encoder.classes_):
    indices = np.where(np.array(loo_true) == i)[0]
    if len(indices) > 0:
        correct = np.sum(np.array(loo_predictions)[indices] == i)
        accuracy = correct / len(indices)
        class_accuracies[driver_id] = accuracy
        print(f"  • {driver_names[driver_id]}: {correct}/{len(indices)} correct ({accuracy:.1%})")

# Check if accuracy is above 85%
if best_accuracy >= 0.85:
    print(f"\n🎉 SUCCESS: Model achieved {best_accuracy:.1%} accuracy (≥85% target)!")
    model_status = "READY_FOR_DEPLOYMENT"
else:
    print(f"\n⚠️ WARNING: Model accuracy is {best_accuracy:.1%} (below 85% target)")
    print("   Recommendations:")
    print("   1. Add more training images per driver")
    print("   2. Ensure good lighting in images")
    print("   3. Capture images from different angles")
    model_status = "NEEDS_IMPROVEMENT"

# Save models
print("\n💾 Saving models...")

model_path = os.path.join(models_dir, "driver_model.pkl")
encoder_path = os.path.join(models_dir, "driver_encoder.pkl")
scaler_path = os.path.join(models_dir, "driver_scaler.pkl")

joblib.dump(best_model, model_path)
joblib.dump(encoder, encoder_path)
joblib.dump(scaler, scaler_path)

# Save comprehensive driver mapping
driver_mapping = {
    'driver_names': driver_names,
    'encoder_classes': encoder.classes_.tolist(),
    'encoder_mapping': {str(cls): i for i, cls in enumerate(encoder.classes_)},
    'processed_counts': processed_counts,
    'n_drivers': len(valid_drivers),
    'n_samples': len(X),
    'cv_accuracy': float(best_accuracy),
    'class_accuracies': {driver_id: float(acc) for driver_id, acc in class_accuracies.items()},
    'training_date': datetime.now().isoformat(),
    'model_type': best_model_name,
    'feature_count': X.shape[1],
    'cv_scores': cv_scores,
    'model_status': model_status,
    'target_accuracy': 0.85,
    'requirements': {
        'min_images_per_driver': 3,
        'min_drivers': 2,
        'feature_count': 50,
        'validation_method': 'Leave-One-Out Cross Validation'
    }
}

mapping_path = os.path.join(models_dir, "driver_mapping.json")
with open(mapping_path, 'w') as f:
    json.dump(driver_mapping, f, indent=2, default=str)

# Generate training report
print(f"\n✅ Models saved successfully:")
print(f"   • Model: {model_path}")
print(f"   • Encoder: {encoder_path}")
print(f"   • Scaler: {scaler_path}")
print(f"   • Mapping: {mapping_path}")
print(f"   • Confusion Matrix: {cm_path}")

print("\n📋 Training Summary:")
print(f"   • Drivers trained: {len(valid_drivers)}")
print(f"   • Total samples: {len(X)}")
print(f"   • Model accuracy: {best_accuracy:.1%}")
print(f"   • Feature count: {X.shape[1]}")
print(f"   • Model status: {model_status}")

if best_accuracy >= 0.85:
    print("\n🎯 MODEL MEETS ACCURACY REQUIREMENTS!")
    print("   The model is ready for deployment in the drowsiness detection system.")
else:
    print("\n⚠️ MODEL NEEDS IMPROVEMENT")
    print("   Please add more training images before deploying.")

# Visualization of feature importance (for RandomForest)
if best_model_name == 'RandomForest':
    print("\n🔍 Feature Importance Analysis:")
    importances = best_model.feature_importances_
    top_features = np.argsort(importances)[-10:][::-1]  # Top 10 features
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), importances[top_features])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Top 10 Most Important Features for Driver Recognition')
    plt.tight_layout()
    fi_path = os.path.join(models_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    print(f"✅ Feature importance plot saved to: {fi_path}")

print("\n" + "="*80)
print("✅ ENHANCED DRIVER RECOGNITION TRAINING COMPLETE!")
print("="*80)

# Show registered drivers for verification
print("\n📋 Registered Drivers (for verification):")
for driver_id, info in valid_drivers.items():
    samples = processed_counts.get(driver_id, 0)
    status = "✅ Ready" if class_accuracies.get(driver_id, 0) >= 0.85 else "⚠️ Needs more data"
    print(f"  • {info['name']} ({driver_id}): {samples} samples - {status}")

print("\n💡 Next steps:")
print("  1. Run the drowsiness detection system")
print("  2. Verify driver recognition in real-time")
print("  3. If accuracy is low, add more training images")
print("  4. Re-train when new drivers are added")