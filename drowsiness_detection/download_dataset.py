import kagglehub
import os
import shutil
import sys
import random
from collections import defaultdict
import re

def download_drowsiness_dataset():
    """Download main drowsiness dataset (contains Drowsy and Non-Drowsy)"""
    print("📥 Downloading Drowsiness Dataset (Drowsy/Non-Drowsy)...")
    try:
        path = kagglehub.dataset_download("ismailnasri20/driver-drowsiness-dataset-ddd")
        print(f"✅ Drowsiness dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading drowsiness dataset: {e}")
        return None

def download_yawning_dataset():
    """Download yawning dataset"""
    print("\n📥 Downloading Yawning Dataset...")
    try:
        path = kagglehub.dataset_download("davidvazquezcic/yawn-dataset")
        print(f"✅ Yawning dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading yawning dataset: {e}")
        return None

def clean_mixed_yawning_dataset(dataset_path):
    """Clean up mixed yawning dataset by removing 'no yawn' images"""
    print("="*70)
    print("CLEANING MIXED YAWNING DATASET")
    print("="*70)
    
    yawning_folder = os.path.join(dataset_path, "Yawning")
    
    if not os.path.exists(yawning_folder):
        print(f"❌ Yawning folder not found: {yawning_folder}")
        return False
    
    # Get all files in Yawning folder
    all_files = [f for f in os.listdir(yawning_folder) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    print(f"📊 Found {len(all_files)} files in Yawning folder")
    
    if len(all_files) == 0:
        print("✅ Yawning folder is already empty")
        return True
    
    # Identify non-yawning files
    non_yawning_files = []
    yawning_files = []
    ambiguous_files = []
    
    for file in all_files:
        file_lower = file.lower()
        
        # Check for indicators of non-yawning
        is_non_yawning = (
            ('no' in file_lower and 'yawn' in file_lower) or  # "no yawn" or "no_yawn"
            ('non' in file_lower and 'yawn' in file_lower) or  # "non_yawn"
            'closed' in file_lower or  # closed mouth
            'normal' in file_lower  # normal face
        )
        
        # Check for indicators of yawning
        is_yawning = (
            'yawn' in file_lower and 
            'no' not in file_lower and 
            'non' not in file_lower and
            not ('closed' in file_lower and 'yawn' in file_lower)
        )
        
        if is_non_yawning:
            non_yawning_files.append(file)
        elif is_yawning:
            yawning_files.append(file)
        else:
            ambiguous_files.append(file)
    
    print(f"\n📊 Analysis:")
    print(f"  Confirmed yawning files: {len(yawning_files)}")
    print(f"  Non-yawning files to remove: {len(non_yawning_files)}")
    print(f"  Ambiguous files: {len(ambiguous_files)}")
    
    # Show samples
    if non_yawning_files:
        print(f"\n🔍 Sample non-yawning files to remove:")
        for file in non_yawning_files[:5]:
            print(f"  - {file}")
    
    if ambiguous_files:
        print(f"\n⚠️ Sample ambiguous files (review manually):")
        for file in ambiguous_files[:5]:
            print(f"  - {file}")
    
    # Remove non-yawning files
    if non_yawning_files:
        print(f"\n🗑️ Removing {len(non_yawning_files)} non-yawning files...")
        for file in non_yawning_files:
            file_path = os.path.join(yawning_folder, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"  ⚠️ Error removing {file}: {e}")
        print(f"✅ Removed non-yawning files")
    else:
        print(f"\n✅ No non-yawning files found!")
    
    # Handle ambiguous files - keep them for now but flag them
    if ambiguous_files:
        print(f"\n⚠️ {len(ambiguous_files)} ambiguous files kept - review manually")
    
    # Rename remaining files to maintain consistent numbering
    remaining_files = [f for f in os.listdir(yawning_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    print(f"\n📁 Remaining yawning files: {len(remaining_files)}")
    
    if remaining_files:
        # Sort files to maintain order
        remaining_files.sort()
        
        print(f"🔢 Re-numbering files...")
        for i, file in enumerate(remaining_files, 1):
            old_path = os.path.join(yawning_folder, file)
            ext = os.path.splitext(file)[1]
            new_name = f"yawning_{i:04d}{ext}"
            new_path = os.path.join(yawning_folder, new_name)
            
            if old_path != new_path:
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f"  ⚠️ Error renaming {file}: {e}")
    
    print(f"✅ Cleanup complete!")
    print(f"   Yawning folder now has {len(remaining_files)} images")
    
    return True

def analyze_yawning_dataset_corrected(yawning_path):
    """CORRECTED: Analyze and extract ONLY yawning images from yawning dataset"""
    print("\n🎯 Analyzing Yawning Dataset (CORRECTED)...")
    
    if not yawning_path:
        print("  No yawning dataset path provided")
        return []
    
    yawning_images = []
    
    for root, dirs, files in os.walk(yawning_path):
        folder_name = os.path.basename(root).lower()
        
        # ONLY take images from "yawn" folder, NOT from "no yawn"
        if 'yawn' in folder_name and 'no' not in folder_name:
            print(f"  📁 Found 'yawn' folder: {folder_name}")
            
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    img_path = os.path.join(root, file)
                    yawning_images.append(img_path)
        
        # Skip "no yawn" folder
        elif 'no yawn' in folder_name or 'no_yawn' in folder_name:
            print(f"  ⚠️ Skipping 'no yawn' folder: {folder_name}")
            continue
    
    print(f"  Found {len(yawning_images)} ACTUAL yawning images")
    
    # Verify we got the right images
    if yawning_images:
        print(f"  Sample verification:")
        for i, img_path in enumerate(yawning_images[:3]):
            print(f"    {i+1}. {os.path.basename(img_path)}")
    
    return yawning_images

def explore_dataset_structure(dataset_path, dataset_name="Dataset"):
    """Explore and understand the dataset structure"""
    print(f"\n🔍 Exploring {dataset_name} structure...")
    
    structure = {
        'folders': [],
        'total_images': 0,
        'file_extensions': defaultdict(int)
    }
    
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = '  ' * level
        
        # Count image files
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        if image_files:
            folder_name = os.path.basename(root)
            print(f"{indent}📁 {folder_name}/ - {len(image_files)} images")
            structure['folders'].append({
                'path': root,
                'name': folder_name,
                'image_count': len(image_files),
                'sample_files': image_files[:3]
            })
            structure['total_images'] += len(image_files)
            
            # Count file extensions
            for file in image_files:
                ext = os.path.splitext(file)[1].lower()
                structure['file_extensions'][ext] += 1
            
        elif dirs and level == 0:
            print(f"{indent}📁 {os.path.basename(root)}/")
    
    print(f"\n📊 {dataset_name} Summary:")
    print(f"  Total images: {structure['total_images']}")
    print(f"  Folder count: {len(structure['folders'])}")
    print(f"  File extensions: {dict(structure['file_extensions'])}")
    
    return structure

def analyze_drowsiness_dataset_structure(dataset_path):
    """Analyze the folder structure to understand organization"""
    print("\n🎯 Analyzing Drowsiness Dataset Structure...")
    
    # Look for common organizational patterns
    folder_stats = defaultdict(list)
    
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root).lower()
        parent_folder = os.path.basename(os.path.dirname(root)).lower()
        
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        if image_files:
            # Try to infer class from folder hierarchy
            if 'drowsy' in folder_name or 'sleep' in folder_name or 'closed' in folder_name:
                folder_stats['drowsy_folders'].append({
                    'path': root,
                    'count': len(image_files),
                    'name': folder_name
                })
            elif 'non' in folder_name or 'alert' in folder_name or 'awake' in folder_name or 'open' in folder_name:
                folder_stats['non_drowsy_folders'].append({
                    'path': root,
                    'count': len(image_files),
                    'name': folder_name
                })
            elif 'yawn' in folder_name or 'yawning' in folder_name:
                folder_stats['yawning_folders'].append({
                    'path': root,
                    'count': len(image_files),
                    'name': folder_name
                })
            else:
                folder_stats['other_folders'].append({
                    'path': root,
                    'count': len(image_files),
                    'name': folder_name
                })
    
    # Print analysis
    print(f"📊 Folder Analysis:")
    print(f"  Drowsy-like folders: {len(folder_stats['drowsy_folders'])}")
    for folder in folder_stats['drowsy_folders'][:3]:  # Show first 3
        print(f"    - {folder['name']}: {folder['count']} images")
    
    print(f"  Non-drowsy-like folders: {len(folder_stats['non_drowsy_folders'])}")
    for folder in folder_stats['non_drowsy_folders'][:3]:
        print(f"    - {folder['name']}: {folder['count']} images")
    
    print(f"  Yawning-like folders: {len(folder_stats['yawning_folders'])}")
    for folder in folder_stats['yawning_folders'][:3]:
        print(f"    - {folder['name']}: {folder['count']} images")
    
    print(f"  Other folders: {len(folder_stats['other_folders'])}")
    
    return folder_stats

def classify_images_by_folder(dataset_path):
    """Classify images based on their folder names"""
    print("\n🎯 Classifying images by folder names...")
    
    classified = {
        'Drowsy': [],
        'Non_Drowsy': [],
        'Yawning': [],
        'Unknown': []
    }
    
    # Expanded keyword lists
    drowsy_keywords = [
        'drowsy', 'sleep', 'closed', 'closed_eye', 'closedeyes',
        'fatigue', 'tired', 'sleepy', 'drowsiness', 'close',
        'drwsy', 'drows', 'drwosy', 'drwozy'  # Common typos
    ]
    
    non_drowsy_keywords = [
        'non', 'non_drowsy', 'non-drowsy', 'alert', 'awake',
        'open', 'open_eye', 'opened', 'normal', 'active',
        'awaken', 'openeyes', 'no_yawn', 'not_drowsy',
        'nondrowsy', 'non drowsy', 'non-drowsiness',
        'awake', 'alertness', 'vigilant'
    ]
    
    yawning_keywords = [
        'yawn', 'yawning', 'mouth', 'open_mouth',
        'mouth_open', 'yawn_', 'yawning_', 'yawns'
    ]
    
    folder_classification = {}
    
    # First pass: classify folders
    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root).lower()
        
        # Check each keyword category
        is_drowsy = any(keyword in folder_name for keyword in drowsy_keywords)
        is_non_drowsy = any(keyword in folder_name for keyword in non_drowsy_keywords)
        is_yawning = any(keyword in folder_name for keyword in yawning_keywords)
        
        # Determine class (prioritize non-drowsy if both present)
        if is_non_drowsy:
            folder_class = 'Non_Drowsy'
        elif is_drowsy:
            folder_class = 'Drowsy'
        elif is_yawning:
            folder_class = 'Yawning'
        else:
            folder_class = 'Unknown'
        
        # Collect images from this folder
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        if image_files and folder_class != 'Unknown':
            print(f"  📁 {folder_name} → {folder_class} ({len(image_files)} images)")
            
            for file in image_files:
                img_path = os.path.join(root, file)
                classified[folder_class].append(img_path)
        elif image_files:
            for file in image_files:
                img_path = os.path.join(root, file)
                classified['Unknown'].append(img_path)
    
    # Print results
    print(f"\n📊 Classification Results:")
    print(f"  Drowsy images: {len(classified['Drowsy'])}")
    print(f"  Non-Drowsy images: {len(classified['Non_Drowsy'])}")
    print(f"  Yawning images: {len(classified['Yawning'])}")
    print(f"  Unknown images: {len(classified['Unknown'])}")
    
    return classified

def create_dataset_structure(target_base):
    """Create the organized dataset folder structure"""
    print(f"\n📁 Creating dataset structure at: {target_base}")
    
    # Remove old dataset if exists
    if os.path.exists(target_base):
        print(f"🗑️ Removing old dataset...")
        shutil.rmtree(target_base)
    
    # Create folders
    classes = ['Drowsy', 'Non_Drowsy', 'Yawning']
    folders = {}
    
    for class_name in classes:
        folder_path = os.path.join(target_base, class_name)
        os.makedirs(folder_path, exist_ok=True)
        folders[class_name] = folder_path
        print(f"  Created: {class_name}/")
    
    return folders

def copy_images_to_folders(classified, folders, max_per_class=500):
    """Copy classified images to their respective folders"""
    print("\n" + "="*50)
    print("COPYING IMAGES TO ORGANIZED FOLDERS")
    print("="*50)
    
    counts = {}
    
    for class_name in ['Drowsy', 'Non_Drowsy', 'Yawning']:
        if class_name in classified and classified[class_name]:
            images = classified[class_name]
            target_folder = folders[class_name]
            
            print(f"\n📂 Processing {class_name} images...")
            print(f"  Found {len(images)} candidate images")
            
            # Limit to max_per_class if needed
            if len(images) > max_per_class:
                selected_images = random.sample(images, max_per_class)
                print(f"  Selecting {max_per_class} random images")
            else:
                selected_images = images
                print(f"  Using all {len(images)} images")
            
            # Copy images
            copied_count = 0
            for i, img_path in enumerate(selected_images):
                try:
                    ext = os.path.splitext(img_path)[1]
                    new_filename = f"{class_name.lower()}_{i+1:04d}{ext}"
                    dst = os.path.join(target_folder, new_filename)
                    shutil.copy2(img_path, dst)
                    copied_count += 1
                except Exception as e:
                    print(f"  ⚠️ Error copying {img_path}: {e}")
            
            counts[class_name] = copied_count
            print(f"  ✅ Copied {copied_count} images to {class_name}/")
        else:
            print(f"\n⚠️ No images found for {class_name}")
            counts[class_name] = 0
    
    return counts

def balance_final_dataset(folders, target_count=500):
    """Balance the final dataset to have equal number of images per class"""
    print("\n" + "="*50)
    print("BALANCING FINAL DATASET")
    print("="*50)
    
    final_counts = {}
    
    for class_name, folder_path in folders.items():
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            
            current_count = len(images)
            
            if current_count > target_count:
                print(f"  {class_name}: Reducing from {current_count} to {target_count}")
                # Remove random excess images
                images_to_remove = random.sample(images, current_count - target_count)
                for img in images_to_remove:
                    os.remove(os.path.join(folder_path, img))
                final_counts[class_name] = target_count
            elif current_count < target_count:
                print(f"  ⚠️ {class_name}: Only {current_count} images (need {target_count})")
                final_counts[class_name] = current_count
            else:
                print(f"  ✅ {class_name}: {current_count} images (perfect)")
                final_counts[class_name] = current_count
        else:
            print(f"  ❌ {class_name}: Folder not found")
            final_counts[class_name] = 0
    
    return final_counts

def print_final_summary(folders):
    """Print final dataset summary"""
    print("\n" + "="*70)
    print("FINAL DATASET SUMMARY")
    print("="*70)
    
    total_images = 0
    
    for class_name in ['Drowsy', 'Non_Drowsy', 'Yawning']:
        folder_path = folders[class_name]
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            count = len(images)
            total_images += count
            
            status = "✅" if count >= 500 else "⚠️" if count >= 300 else "❌"
            print(f"{status} {class_name}: {count} images")
            
            # Show sample filenames
            if images:
                print(f"     Sample: {', '.join(images[:3])}")
        else:
            print(f"❌ {class_name}: Folder not found")
    
    print(f"\n📊 Total images: {total_images}")
    
    # Check if ready for training
    if total_images >= 1500:
        print("\n🎉 Dataset is complete and ready for training!")
        print("   Run: python train_drowsiness.py")
    elif total_images >= 900:
        print("\n⚠️ Dataset has minimum images for training")
        print("   Consider collecting more images for better performance")
        print("   Run: python train_drowsiness.py")
    else:
        print("\n❌ Dataset needs more images")
        print("   Some classes have insufficient samples")

def organize_all_datasets():
    """Main function to organize all datasets with corrected yawning handling"""
    print("="*70)
    print("COMPLETE DROWSINESS DATASET ORGANIZER (CORRECTED)")
    print("="*70)
    
    # Set target location
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_base = os.path.join(project_root, "datasets", "complete_drowsiness")
    
    print(f"\n🎯 Target location: {target_base}")
    
    # First, clean up any existing mixed dataset
    if os.path.exists(target_base):
        print("\n" + "="*50)
        print("CLEANING EXISTING DATASET")
        print("="*50)
        clean_mixed_yawning_dataset(target_base)
    
    # Download datasets
    print("\n" + "="*50)
    print("DOWNLOADING DATASETS")
    print("="*50)
    
    # Download drowsiness dataset
    drowsiness_path = download_drowsiness_dataset()
    if not drowsiness_path:
        print("❌ Failed to download drowsiness dataset. Exiting.")
        return None
    
    # Download yawning dataset
    yawning_path = download_yawning_dataset()
    
    # Explore dataset structures
    print("\n" + "="*50)
    print("EXPLORING DATASET STRUCTURES")
    print("="*50)
    
    explore_dataset_structure(drowsiness_path, "Drowsiness Dataset")
    
    if yawning_path:
        explore_dataset_structure(yawning_path, "Yawning Dataset")
    
    # Analyze folder structure to understand organization
    folder_stats = analyze_drowsiness_dataset_structure(drowsiness_path)
    
    # Classify images from drowsiness dataset
    print("\n" + "="*50)
    print("CLASSIFYING DROWSINESS DATASET IMAGES")
    print("="*50)
    
    classified = classify_images_by_folder(drowsiness_path)
    
    # Get ONLY yawning images from yawning dataset (not "no yawn")
    if yawning_path:
        yawning_images = analyze_yawning_dataset_corrected(yawning_path)
        if yawning_images:
            # Add to classified yawning list
            if 'Yawning' not in classified:
                classified['Yawning'] = []
            classified['Yawning'].extend(yawning_images)
            print(f"  ✅ Added {len(yawning_images)} ACTUAL yawning images from yawning dataset")
        else:
            print("  ⚠️ No yawning images found in yawning dataset!")
    
    # Create folder structure
    folders = create_dataset_structure(target_base)
    
    # Copy images to organized folders
    counts = copy_images_to_folders(classified, folders, max_per_class=500)
    
    # Balance final dataset
    final_counts = balance_final_dataset(folders, target_count=500)
    
    # Verify yawning dataset one more time
    print("\n" + "="*50)
    print("FINAL YAWNING DATASET VERIFICATION")
    print("="*50)
    
    yawning_folder = folders['Yawning']
    if os.path.exists(yawning_folder):
        yawning_files = os.listdir(yawning_folder)
        print(f"📊 Yawning folder contents: {len(yawning_files)} files")
        
        # Check for any remaining "no yawn" files
        no_yawn_files = [f for f in yawning_files if 'no' in f.lower()]
        if no_yawn_files:
            print(f"⚠️ Found {len(no_yawn_files)} files with 'no' in filename")
            print("  Removing them...")
            for file in no_yawn_files:
                os.remove(os.path.join(yawning_folder, file))
            print(f"  Removed {len(no_yawn_files)} files")
        
        # Count remaining files
        remaining = len([f for f in os.listdir(yawning_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
        print(f"✅ Final yawning count: {remaining} images")
    
    # Final summary
    print_final_summary(folders)
    
    return target_base

def clean_existing_dataset_only():
    """Only clean the existing dataset without re-downloading"""
    print("="*70)
    print("CLEANING EXISTING DATASET ONLY")
    print("="*70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "datasets", "complete_drowsiness")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Running full organization instead...")
        return organize_all_datasets()
    
    print(f"Found existing dataset at: {dataset_path}")
    
    # Clean the yawning dataset
    clean_mixed_yawning_dataset(dataset_path)
    
    # Print final status
    print("\n" + "="*70)
    print("CLEANUP COMPLETE!")
    print("="*70)
    
    total_images = 0
    for class_name in ['Drowsy', 'Non_Drowsy', 'Yawning']:
        folder_path = os.path.join(dataset_path, class_name)
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            count = len(images)
            total_images += count
            status = "✅" if count >= 500 else "⚠️" if count >= 300 else "❌"
            print(f"{status} {class_name}: {count} images")
        else:
            print(f"❌ {class_name}: Folder not found")
    
    print(f"\n📊 Total images: {total_images}")
    
    if total_images >= 1500:
        print("\n🎉 Dataset is ready for training!")
    else:
        print("\n⚠️ Dataset needs more images")
    
    return dataset_path

def main():
    """Main execution function - automatically cleans existing dataset"""
    print("="*70)
    print("AUTOMATIC DATASET CLEANER")
    print("="*70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "datasets", "complete_drowsiness")
    
    # Check if dataset exists
    if os.path.exists(dataset_path):
        print(f"✅ Found existing dataset at: {dataset_path}")
        
        # Check if Yawning folder exists and has files
        yawning_folder = os.path.join(dataset_path, "Yawning")
        if os.path.exists(yawning_folder):
            yawning_files = [f for f in os.listdir(yawning_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            
            if yawning_files:
                print(f"📊 Yawning folder has {len(yawning_files)} files")
                
                # Check if any "no yawn" files exist
                no_yawn_files = [f for f in yawning_files if 'no' in f.lower()]
                if no_yawn_files:
                    print(f"⚠️ Found {len(no_yawn_files)} 'no yawn' files - cleaning...")
                    # Run the cleaner
                    clean_existing_dataset_only()
                else:
                    print("✅ Yawning folder appears clean - no 'no yawn' files found")
                    print("🎉 Dataset is ready for training!")
            else:
                print("⚠️ Yawning folder is empty - running full organization...")
                organize_all_datasets()
        else:
            print("⚠️ Yawning folder not found - running full organization...")
            organize_all_datasets()
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Running full organization...")
        organize_all_datasets()

if __name__ == "__main__":
    main()