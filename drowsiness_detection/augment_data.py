# augment_data.py - Data augmentation for drowsiness detection
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def augment_images(input_folder, output_folder, target_count=800):
    """Augment images to reach target count"""
    
    # Get all images
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not images:
        print(f"No images found in {input_folder}")
        return
    
    current_count = len(images)
    needed = target_count - current_count
    
    if needed <= 0:
        print(f"Already have {current_count} images, target is {target_count}")
        return
    
    print(f"Augmenting {current_count} images to {target_count} (+{needed})")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy original images
    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        output_path = os.path.join(output_folder, f"original_{i:04d}.jpg")
        cv2.imwrite(output_path, img)
    
    # Generate augmented images
    augment_idx = 0
    pbar = tqdm(total=needed, desc="Augmenting images")
    
    while augment_idx < needed:
        for img_path in images:
            if augment_idx >= needed:
                break
                
            img = cv2.imread(img_path)
            
            # Apply different augmentations
            for aug_type in range(5):  # 5 different augmentations per image
                if augment_idx >= needed:
                    break
                    
                augmented = img.copy()
                
                # Random rotation
                angle = np.random.uniform(-30, 30)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                augmented = cv2.warpAffine(augmented, M, (w, h))
                
                # Random brightness
                brightness = np.random.uniform(0.7, 1.3)
                augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
                
                # Random contrast
                contrast = np.random.uniform(0.7, 1.3)
                augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=0)
                
                # Random flip
                if np.random.random() > 0.5:
                    augmented = cv2.flip(augmented, 1)
                
                # Add noise
                if np.random.random() > 0.7:
                    noise = np.random.normal(0, 15, augmented.shape).astype(np.uint8)
                    augmented = cv2.add(augmented, noise)
                
                # Save augmented image
                output_path = os.path.join(output_folder, f"aug_{augment_idx:04d}.jpg")
                cv2.imwrite(output_path, augmented)
                
                augment_idx += 1
                pbar.update(1)
    
    pbar.close()
    print(f"✅ Augmentation complete. Total images: {current_count + needed}")

if __name__ == "__main__":
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_path = os.path.join(project_root, "datasets", "complete_drowsiness")
    
    # Augment each class
    classes = ["Drowsy", "Non_Drowsy", "Yawning"]
    
    for cls in classes:
        print(f"\n{'='*50}")
        print(f"Augmenting {cls} class")
        print(f"{'='*50}")
        
        input_folder = os.path.join(dataset_path, cls)
        augment_images(input_folder, input_folder, target_count=800)
    
    print("\n" + "="*50)
    print("✅ DATA AUGMENTATION COMPLETE!")
    print("="*50)
    print("\nNow run train_drowsiness.py again with the augmented dataset.")