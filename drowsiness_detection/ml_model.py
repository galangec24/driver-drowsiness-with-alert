# ml_model.py - UPDATED WITH num_features ATTRIBUTE
import os
import numpy as np
import json

class DrowsinessMLModel:
    def __init__(self, models_dir='../models'):
        """
        Initialize ML model for drowsiness detection
        
        Args:
            models_dir: Path to models directory (default: ../models)
        """
        print(f"📁 Looking for models in: {os.path.abspath(models_dir)}")
        
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.is_available = False
        self.accuracy = 0.0
        self.thresholds = {}
        self.num_features = 56  # Default for 39-feature model
        
        # Feature structure (39 features total)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [13, 14, 78, 308, 17, 18]
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained drowsiness ML model"""
        try:
            import joblib
            
            model_path = os.path.join(self.models_dir, 'drowsiness_model.pkl')
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            accuracy_path = os.path.join(self.models_dir, 'accuracy_report.json')
            thresholds_path = os.path.join(self.models_dir, 'thresholds.txt')
            
            # Check if files exist
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print(f"✅ Found model files in: {os.path.abspath(self.models_dir)}")
                
                # Load model and scaler
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_available = True
                print(f"✅ ML model loaded successfully!")
                
                # Get actual number of features from scaler
                if hasattr(self.scaler, 'mean_'):
                    self.num_features = len(self.scaler.mean_)
                    print(f"📊 Model expects {self.num_features} features")
                    
                    if self.num_features == 39:
                        print("✅ Model configured for 39 features (matching training)")
                    else:
                        print(f"⚠️ Model expects {self.num_features} features, not 39")
                        print("   This may indicate a mismatch with training data")
                else:
                    print(f"⚠️ Could not determine feature count from scaler")
                    print(f"   Using default: {self.num_features} features")
                
                # Load accuracy report
                if os.path.exists(accuracy_path):
                    with open(accuracy_path, 'r') as f:
                        accuracy_data = json.load(f)
                    self.accuracy = accuracy_data.get('accuracy', 0.0)
                    print(f"📈 Training Accuracy: {self.accuracy:.1%}")
                else:
                    print("⚠️ Accuracy report not found")
                
                # Load thresholds
                if os.path.exists(thresholds_path):
                    try:
                        with open(thresholds_path, 'r') as f:
                            thresholds_content = f.read()
                        
                        # Parse thresholds
                        for line in thresholds_content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                if '=' in line:
                                    key, value = line.split('=', 1)
                                elif ':' in line:
                                    key, value = line.split(':', 1)
                                else:
                                    continue
                                key = key.strip()
                                value = value.strip()
                                try:
                                    self.thresholds[key] = float(value)
                                except ValueError:
                                    pass
                        
                        if self.thresholds:
                            print("📊 Loaded model thresholds:")
                            for key, value in self.thresholds.items():
                                print(f"   • {key}: {value}")
                        else:
                            print("⚠️ No thresholds loaded from file")
                    except Exception as e:
                        print(f"⚠️ Error reading thresholds: {e}")
                else:
                    print("⚠️ Thresholds file not found")
                
            else:
                print(f"❌ ML model files not found in: {self.models_dir}")
                print("💡 Looking for:")
                print(f"   • {model_path}")
                print(f"   • {scaler_path}")
                
                # List what's actually in the directory
                if os.path.exists(self.models_dir):
                    print("\n📁 Files found in models directory:")
                    for file in os.listdir(self.models_dir):
                        print(f"   • {file}")
                else:
                    print(f"\n❌ Directory does not exist: {self.models_dir}")
                
        except ImportError:
            print("⚠️ Joblib not available - ML features disabled")
            print("💡 Install with: pip install joblib")
        except Exception as e:
            print(f"⚠️ Error loading ML model: {e}")
            import traceback
            traceback.print_exc()
    
    def create_feature_vector(self, left_eye_points, right_eye_points, 
                        mouth_points, ear, mar, eye_distance, eye_asymmetry=0.0):
        """
        Create the exact feature vector that matches training data (56 features)
        """
        features = []
        
        # Add left eye coordinates (12 features)
        for x, y in left_eye_points:
            features.extend([x, y])
        
        # Add right eye coordinates (12 features)
        for x, y in right_eye_points:
            features.extend([x, y])
        
        # Add mouth coordinates (12 features)
        for x, y in mouth_points:
            features.extend([x, y])
        
        # Add face contour coordinates (8 features) - placeholder for now
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add eyebrow coordinates (8 features) - placeholder for now
        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add calculated features (4 features)
        features.extend([ear, mar, eye_distance, eye_asymmetry])
        
        # Check if we have the right number of features
        if len(features) != self.num_features:
            print(f"⚠️ Feature mismatch: Created {len(features)} features, expected {self.num_features}")
            
            # For now, pad or truncate to match
            if len(features) < self.num_features:
                features.extend([0.0] * (self.num_features - len(features)))
            else:
                features = features[:self.num_features]
        
        return np.array(features)
    
    def predict_drowsiness(self, left_eye_points, right_eye_points,
                      mouth_points, ear, mar, eye_distance, eye_asymmetry=0.0):
        """
        Predict drowsiness using the full 56-feature model
        """
        if not self.is_available or self.model is None or self.scaler is None:
            return 0.0, 0.0
        
        try:
            # Create feature vector
            features = self.create_feature_vector(
                left_eye_points, right_eye_points, 
                mouth_points, ear, mar, eye_distance, eye_asymmetry
            )
            
            # Reshape for prediction
            features_reshaped = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features_reshaped)
            
            # Predict
            probability = self.model.predict_proba(features_scaled)[0]
            
            # Handle different number of classes
            if len(probability) > 1:
                # Multi-class: probability[0] = NORMAL, probability[1] = DROWSY, probability[2] = YAWNING
                drowsy_prob = probability[1] if len(probability) > 1 else 0.0
            else:
                # Single class: probability is NORMAL
                drowsy_prob = 1.0 - probability[0]
            
            confidence = min(1.0, drowsy_prob * 1.2)
            
            return drowsy_prob, confidence
            
        except Exception as e:
            print(f"⚠️ Error in ML prediction: {e}")
            return 0.0, 0.0