# train_drowsiness.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from collections import Counter
import cv2

print("="*80)
print("🚀 DROWSINESS DETECTION TRAINING (15 Epochs)")
print("="*80)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, "models")
dataset_path = os.path.join(project_root, "datasets", "complete_drowsiness")

print(f"📁 Models directory: {models_dir}")
print(f"📁 Dataset: {dataset_path}")

# Check dataset
if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found: {dataset_path}")
    print(f"   Please run: python download_dataset.py first")
    exit()

# Check class sizes
class_counts = {}
class_folders = ["Drowsy", "Non_Drowsy", "Yawning"]

for cls in class_folders:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.exists(cls_path):
        images = glob.glob(os.path.join(cls_path, "*.jpg")) + \
                 glob.glob(os.path.join(cls_path, "*.png")) + \
                 glob.glob(os.path.join(cls_path, "*.jpeg"))
        class_counts[cls] = len(images)
        print(f"  {cls}: {len(images)} images")
    else:
        print(f"  ❌ {cls}: Folder not found at {cls_path}")

print("\n⚖️ Class Distribution:")
for cls, count in class_counts.items():
    percentage = (count / sum(class_counts.values())) * 100
    print(f"  {cls}: {count} images ({percentage:.1f}%)")

# Build a simpler but effective model
def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=3):
    """Build MobileNetV2 model - better for smaller datasets"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_simple_cnn(input_shape=(224, 224, 3), num_classes=3):
    """Build a simple CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create powerful data augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.15  # 15% for validation
)

# Create generators
print("\n📊 Creating data generators...")

# Check if images exist before creating generators
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")

# Get class names
class_names = list(train_generator.class_indices.keys())
print(f"✅ Classes: {class_names}")

# Calculate class weights for imbalanced data
print("\n⚖️ Calculating class weights...")
class_counts_train = Counter(train_generator.classes)
total_samples = train_generator.samples
class_weights = {}

for i, (class_name, class_index) in enumerate(train_generator.class_indices.items()):
    if class_index in class_counts_train and class_counts_train[class_index] > 0:
        weight = total_samples / (len(class_names) * class_counts_train[class_index])
        
        # Boost important classes
        if class_name == "Drowsy":
            weight = min(weight * 1.5, 3.0)
        elif class_name == "Yawning":
            weight = min(weight * 1.3, 2.5)
        else:
            weight = min(weight, 2.0)
        
        class_weights[class_index] = weight
        print(f"  {class_name}: weight = {weight:.2f}")

# Build model - try MobileNetV2 first
print("\n🏗️ Building MobileNetV2 model...")
model = build_mobilenet_model()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("\n📊 Model Summary:")
model.summary()

# Create models directory
os.makedirs(models_dir, exist_ok=True)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(models_dir, 'best_drowsiness_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Training
print("\n🔥 Starting training (15 Epochs)...")
try:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights
    )
    print("✅ Training completed successfully!")
    
except Exception as e:
    print(f"❌ Error during training: {e}")
    print("\n⚠️ Trying with Simple CNN model instead...")
    
    # Try with simpler model
    model = build_simple_cnn()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights
    )

# Load best model if available
best_model_path = os.path.join(models_dir, 'best_drowsiness_model.h5')
if os.path.exists(best_model_path):
    print(f"\n📊 Loading best model from: {best_model_path}")
    model = keras.models.load_model(best_model_path)
else:
    print("\n📊 Using final trained model")

# Evaluate
print("\n📊 Final Evaluation:")
results = model.evaluate(validation_generator, verbose=1)

if isinstance(results, list):
    if len(results) >= 5:
        val_loss, val_accuracy, val_precision, val_recall, val_auc = results[:5]
    elif len(results) >= 2:
        val_loss, val_accuracy = results[:2]
        val_precision = val_recall = val_auc = 0
    else:
        val_loss = val_accuracy = val_precision = val_recall = val_auc = 0
else:
    val_loss = val_accuracy = val_precision = val_recall = val_auc = 0

print(f"  • Validation Loss: {val_loss:.4f}")
print(f"  • Validation Accuracy: {val_accuracy:.2%}")
print(f"  • Validation Precision: {val_precision:.2%}")
print(f"  • Validation Recall: {val_recall:.2%}")
print(f"  • Validation AUC: {val_auc:.2%}")

# Generate predictions
print("\n📈 Generating predictions...")
validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Classification report
print("\n📋 Classification Report:")
report = classification_report(y_true, y_pred_classes, 
                              target_names=class_names, 
                              output_dict=True)
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Focus on Drowsy class
if "Drowsy" in report and isinstance(report["Drowsy"], dict):
    drowsy_precision = report["Drowsy"]["precision"]
    drowsy_recall = report["Drowsy"]["recall"]
    drowsy_f1 = report["Drowsy"]["f1-score"]
else:
    drowsy_precision = drowsy_recall = drowsy_f1 = 0

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(models_dir, 'confusion_matrix.png'), dpi=150)
print(f"✅ Saved confusion matrix to '{os.path.join(models_dir, 'confusion_matrix.png')}'")

# Save model
main_model_path = os.path.join(models_dir, "drowsiness_model.h5")
try:
    model.save(main_model_path)
    print(f"✅ Model saved to: {main_model_path}")
except Exception as e:
    print(f"⚠️ Error saving model: {e}")

# Save training info
class_info = {
    'class_indices': train_generator.class_indices,
    'classes': class_names,
    'training_date': datetime.now().isoformat(),
    'dataset_info': class_counts,
    'performance': {
        'val_accuracy': float(val_accuracy),
        'val_precision': float(val_precision),
        'val_recall': float(val_recall),
        'val_auc': float(val_auc),
        'val_loss': float(val_loss)
    },
    'drowsy_performance': {
        'precision': float(drowsy_precision),
        'recall': float(drowsy_recall),
        'f1_score': float(drowsy_f1)
    }
}

info_path = os.path.join(models_dir, "drowsiness_classes.json")
with open(info_path, 'w') as f:
    json.dump(class_info, f, indent=2)
print(f"✅ Saved class info to: {info_path}")

# Plot training history
if 'history' in locals() and history.history:
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='85% Target')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Training', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, 'training_history.png'), dpi=150)
    print(f"✅ Saved training history to '{os.path.join(models_dir, 'training_history.png')}'")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 Final Results:")
print(f"  • Overall Accuracy: {val_accuracy:.1%}")
print(f"  • Drowsy Recall: {drowsy_recall:.1%}")
print(f"  • Drowsy Precision: {drowsy_precision:.1%}")

# Performance evaluation
if val_accuracy >= 0.85:
    print("\n🎉 EXCELLENT: Target achieved! Accuracy ≥ 85%")
    print("   Model is ready for deployment!")
elif val_accuracy >= 0.75:
    print("\n📈 GOOD: Decent accuracy")
    print("   Suggestions for 85%+:")
    print("   1. Add more training data")
    print("   2. Train for 5-10 more epochs")
    print("   3. Try EfficientNetB0 architecture")
elif val_accuracy >= 0.65:
    print("\n⚠️ FAIR: Model works but needs improvement")
    print("   To reach 85%:")
    print("   1. Collect more diverse images")
    print("   2. Check dataset quality")
    print("   3. Adjust augmentation parameters")
else:
    print("\n❌ NEEDS WORK: Significant improvement needed")
    print("   Critical actions:")
    print("   1. Verify all images are correctly labeled")
    print("   2. Ensure sufficient training data per class")
    print("   3. Try different model architecture")

# Test model on a few validation images
print(f"\n🔍 Testing on {min(3, len(validation_generator.filenames))} validation images:")
validation_generator.reset()
for i in range(min(3, len(validation_generator.filenames))):
    batch_x, batch_y = validation_generator.next()
    pred = model.predict(batch_x, verbose=0)
    true_class = class_names[np.argmax(batch_y[0])]
    pred_class = class_names[np.argmax(pred[0])]
    confidence = np.max(pred[0])
    print(f"  Image {i+1}: True={true_class}, Pred={pred_class}, Conf={confidence:.1%}")

print(f"\n📁 Model and results saved in: {models_dir}")
print("="*80)   