# check_structure.py
import os

def check_project_structure():
    print("🔍 Checking project structure...")
    print("="*60)
    
    # Current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in drowsiness_detection folder
    folder_name = os.path.basename(current_dir)
    print(f"Folder name: {folder_name}")
    
    # Check for important files
    important_files = [
        'drowsiness_detector.py',
        'ml_model.py', 
        'detector.py',
        'face_processor.py',
        'utils.py',
        'dashboard.py',
        'app.py',
        'train_model.py'
    ]
    
    print("\n📁 Files in current directory:")
    for file in important_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"  {status} {file}")
    
    # Check for models directory
    print("\n🔍 Looking for models directory:")
    possible_model_paths = [
        'models',
        '../models',
        '../../models',
        './backend/models',
        '../backend/models',
        os.path.join(current_dir, 'models'),
        os.path.join(current_dir, '..', 'models'),
    ]
    
    for path in possible_model_paths:
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
            print(f"✅ Found: {path} -> {abs_path}")
            
            # List model files
            if os.path.isdir(path):
                print(f"   Files in {path}:")
                for file in os.listdir(path):
                    print(f"     • {file}")
        else:
            print(f"❌ Not found: {path}")
    
    # Check for parent directories
    print("\n📁 Parent directory structure:")
    parent_dir = os.path.dirname(current_dir)
    if os.path.exists(parent_dir):
        print(f"Parent: {parent_dir}")
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    
    print("\n💡 Recommendation:")
    print("1. Make sure your models folder contains:")
    print("   • drowsiness_model.pkl")
    print("   • scaler.pkl")
    print("   • thresholds.txt")
    print("   • accuracy_report.json")
    print("\n2. If models are in a different location, update ml_model.py")
    print("   or pass the correct path when creating DrowsinessMLModel")
    print("="*60)

if __name__ == "__main__":
    check_project_structure()