#!/usr/bin/env python3
"""
Main driver drowsiness detection system
Uses 39-feature ML model for accurate drowsiness prediction
"""

import os
import warnings
warnings.filterwarnings('ignore')

from detector import AdvancedDrowsinessDetector

def main():
    print("\n" + "="*80)
    print("🚗 ADVANCED DRIVER DROWSINESS MONITORING SYSTEM")
    print("="*80)
    
    # Check for models
    models_path = '../models'
    model_file = os.path.join(models_path, 'drowsiness_model.pkl')
    
    if os.path.exists(model_file):
        print(f"✅ Model found at: {os.path.abspath(model_file)}")
    else:
        print(f"❌ Model not found at: {os.path.abspath(model_file)}")
        print("💡 Please ensure models are in the ../models folder")
        return
    
    try:
        # Create detector with correct models path
        detector = AdvancedDrowsinessDetector()
        detector.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thank you for using the Driver Drowsiness Monitoring System!")

if __name__ == "__main__":
    main()