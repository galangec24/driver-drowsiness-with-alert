"""
Dashboard visualization functions with integrated driver recognition - UPDATED
"""

import cv2
import numpy as np
import sqlite3
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import time
import json

class RealTimeDriverClassifier:
    def __init__(self, model_path='../models/driver_classifier.pkl', db_path='../backend/drivers.db'):
        self.model_path = model_path
        self.db_path = db_path
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Driver database cache
        self.driver_names = {}
        self.load_driver_names()
        
        # Load or create model - FIXED with better error handling
        self.model = None
        self.label_encoder = None
        self.model_loaded = False
        
        # Try to load the model
        self.load_classifier_model()
        
        # Recognition tracking
        self.last_recognition_time = 0
        self.recognition_interval = 3  # Recognize every 3 seconds
        self.current_driver_name = "Unknown"
        self.current_driver_id = None
        self.current_confidence = 0.0
        self.recognition_count = 0
    
    def load_driver_names(self):
        """Load driver names from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT driver_id, name FROM drivers')
            drivers = cursor.fetchall()
            conn.close()
            
            for driver_id, name in drivers:
                self.driver_names[driver_id] = name
            
            print(f"✓ Loaded {len(self.driver_names)} driver names from database")
            if len(self.driver_names) > 0:
                print(f"   Drivers: {list(self.driver_names.values())}")
        except Exception as e:
            print(f"⚠️ Error loading driver names: {e}")
            # Create a dummy driver for testing
            self.driver_names = {1: "Test Driver"}
    
    def load_classifier_model(self):
        """Load the classifier model with better error handling"""
        print("\n🔍 Loading Driver Classifier Model...")
        
        try:
            # Check if model files exist
            model_exists = os.path.exists(self.model_path)
            encoder_path = self.model_path.replace('classifier.pkl', 'label_encoder.pkl')
            mapping_path = self.model_path.replace('classifier.pkl', 'mapping.json')
            
            print(f"   • Model file: {'✅ Found' if model_exists else '❌ Missing'}")
            print(f"   • Encoder file: {'✅ Found' if os.path.exists(encoder_path) else '❌ Missing'}")
            print(f"   • Mapping file: {'✅ Found' if os.path.exists(mapping_path) else '❌ Missing'}")
            
            if not model_exists:
                print("❌ Model file not found. Training required.")
                self.model_loaded = False
                return
            
            # Load model
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded. Type: {type(self.model).__name__}")
            
            # Try to load encoder (try multiple possible names)
            encoder_files = [
                '../models/driver_label_encoder.pkl',
                '../models/label_encoder.pkl',
                '../models/driver_label_encoder.pkl'
            ]
            
            encoder_loaded = False
            for encoder_file in encoder_files:
                if os.path.exists(encoder_file):
                    self.label_encoder = joblib.load(encoder_file)
                    print(f"✅ Label encoder loaded from: {encoder_file}")
                    print(f"   • Classes: {list(self.label_encoder.classes_)}")
                    encoder_loaded = True
                    break
            
            if not encoder_loaded:
                print("⚠️ Label encoder not found. Creating new one.")
                self.label_encoder = LabelEncoder()
            
            # Load driver mapping
            mapping_files = [
                '../models/driver_mapping.json',
                '../models/mapping.json',
                'driver_mapping.json',
                'mapping.json'
            ]
            
            for mapping_file in mapping_files:
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        mapping = json.load(f)
                    print(f"✅ Driver mapping loaded: {len(mapping)} drivers")
                    break
            
            self.model_loaded = True
            print("✅ Driver classifier ready!")
            
            # Print model details
            if hasattr(self.model, 'n_features_in_'):
                print(f"   • Expected features: {self.model.n_features_in_}")
            if hasattr(self.model, 'classes_'):
                print(f"   • Model classes: {self.model.classes_}")
            
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def extract_face_features(self, frame):
        """Extract facial features from video frame - IMPROVED"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            features = []
            
            # Extract 68 key landmarks for better recognition
            key_indices = [
                # Eyes (12 points)
                33, 133, 157, 158, 159, 160, 161, 162,  # Right eye
                362, 263, 386, 387, 388, 389, 390, 391,  # Left eye
                
                # Eyebrows (8 points)
                105, 107, 66, 70,  # Right eyebrow
                285, 295, 282, 283, # Left eyebrow
                
                # Nose (9 points)
                1, 2, 3, 4, 5, 6, 195, 197,  # Nose bridge and tip
                
                # Mouth (20 points)
                13, 14, 17, 18, 37, 39, 40, 41,  # Outer lips
                61, 62, 78, 80, 81, 82, 87, 88,   # Inner lips
                308, 310, 311, 312,               # Mouth corners
                
                # Face contour (19 points)
                10, 50, 109, 67, 103, 54, 21,  # Chin and jaw
                162, 389, 359, 386, 374, 264,  # Cheeks
                152, 148, 176, 149, 150        # Forehead
            ]
            
            for idx in key_indices:
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    features.extend([landmark.x, landmark.y])
                    # Optional: Add z coordinate for 3D recognition
                    # features.append(landmark.z)
            
            # Ensure consistent feature size
            if len(features) < 136:  # 68 points * 2 coordinates
                features.extend([0.0] * (136 - len(features)))
            elif len(features) > 136:
                features = features[:136]
            
            return np.array(features, dtype=np.float32)
        return None
    
    def recognize_driver(self, frame):
        """Recognize driver from video frame - IMPROVED"""
        self.recognition_count += 1
        
        # If model not loaded, return unknown
        if not self.model_loaded:
            if self.recognition_count % 30 == 0:  # Print every 30 frames
                print("⚠️ Driver classifier not loaded. Training required.")
            return "Unknown (Train Model)", 0.0
        
        current_time = time.time()
        
        # Only recognize at intervals to save CPU
        if current_time - self.last_recognition_time < self.recognition_interval:
            return self.current_driver_name, self.current_confidence
        
        # Extract features
        features = self.extract_face_features(frame)
        if features is None:
            return "No Face", 0.0
        
        try:
            # Reshape for prediction
            features_reshaped = features.reshape(1, -1)
            
            # Check feature count matches model expectation
            if hasattr(self.model, 'n_features_in_'):
                expected_features = self.model.n_features_in_
                if len(features) != expected_features:
                    print(f"⚠️ Feature mismatch: Got {len(features)}, expected {expected_features}")
                    # Try to pad or truncate
                    if len(features) < expected_features:
                        padded = np.zeros(expected_features)
                        padded[:len(features)] = features
                        features_reshaped = padded.reshape(1, -1)
                    else:
                        features_reshaped = features[:expected_features].reshape(1, -1)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_reshaped)[0]
                max_prob = np.max(probabilities)
                pred_index = np.argmax(probabilities)
                
                if self.recognition_count % 10 == 0:  # Log every 10th recognition
                    print(f"🔍 Recognition: max_prob={max_prob:.3f}, index={pred_index}")
                
                if max_prob > 0.5:  # Lower threshold for testing
                    try:
                        driver_id = self.label_encoder.inverse_transform([pred_index])[0]
                        driver_name = self.driver_names.get(driver_id, f"Driver_{driver_id}")
                        
                        self.current_driver_id = driver_id
                        self.current_driver_name = driver_name
                        self.current_confidence = float(max_prob)
                        
                        if self.recognition_count % 10 == 0:
                            print(f"✅ Recognized: {driver_name} (ID: {driver_id}, Confidence: {max_prob:.1%})")
                    except Exception as e:
                        print(f"⚠️ Error in label inversion: {e}")
                        self.current_driver_name = "Unknown"
                        self.current_confidence = float(max_prob)
                else:
                    self.current_driver_name = "Unknown"
                    self.current_confidence = float(max_prob)
            else:
                # Direct prediction (no probabilities)
                prediction = self.model.predict(features_reshaped)[0]
                driver_name = self.driver_names.get(prediction, f"Driver_{prediction}")
                self.current_driver_name = driver_name
                self.current_confidence = 0.7  # Default confidence
            
            self.last_recognition_time = current_time
            return self.current_driver_name, self.current_confidence
            
        except Exception as e:
            print(f"⚠️ Recognition error: {e}")
            return "Unknown (Error)", 0.0

# Initialize global classifier with debug info
print("\n" + "="*60)
print("🚗 INITIALIZING DRIVER RECOGNITION SYSTEM")
print("="*60)
driver_classifier = RealTimeDriverClassifier()
print("="*60)

def create_dashboard1(frame, ear, mar, status, blink_detected, face_detected,
                     driver_name, fps, total_blinks, blink_rate, ml_model, 
                     face_bbox=None):
    """Create Dashboard 1: Live Monitoring - with driver recognition"""
    h, w = frame.shape[:2]
    
    # Status colors
    if not face_detected:
        status_color = (128, 128, 128)  # Gray
    elif "DROWSY" in status:
        status_color = (0, 0, 255)      # Red
    elif "YAWNING" in status:
        status_color = (0, 165, 255)    # Orange
    elif "BLINKING" in status:
        status_color = (255, 255, 0)    # Yellow
    else:
        status_color = (0, 255, 0)      # Green
    
    # === TOP-RIGHT CORNER: FPS only ===
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # === DRIVER RECOGNITION ON FACE BOX ===
    if face_detected and face_bbox:
        x1, y1, x2, y2 = face_bbox
        
        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Recognize driver
        recognized_name, confidence = driver_classifier.recognize_driver(frame)
        
        # Use recognized name if confidence is good
        if recognized_name != "Unknown" and confidence > 0.5:
            display_name = recognized_name
            name_color = (0, 255, 0)  # Green for recognized
            confidence_text = f"{confidence:.0%}"
        else:
            display_name = "Unknown Driver"
            name_color = (255, 165, 0)  # Orange for unknown
            confidence_text = ""
        
        # Show driver name ABOVE the face box
        name_y = max(30, y1 - 10)
        
        # Prepare text
        if "Train Model" in display_name or "Error" in display_name:
            status_text = display_name
            text_color = (0, 165, 255)  # Orange for warnings
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            # Draw warning background
            cv2.rectangle(frame, 
                         (x1 - 5, name_y - text_size[1] - 5),
                         (x1 + text_size[0] + 5, name_y + 5),
                         (0, 50, 100), -1)
            
            cv2.putText(frame, status_text, (x1, name_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        else:
            name_text = f"Driver: {display_name}"
            text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw name background
            cv2.rectangle(frame, 
                         (x1 - 5, name_y - text_size[1] - 5),
                         (x1 + text_size[0] + 5, name_y + 5),
                         (40, 40, 40), -1)
            
            # Draw name
            cv2.putText(frame, name_text, (x1, name_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_color, 2)
            
            # Draw confidence if available
            if confidence_text:
                conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame,
                            (x1 + text_size[0] + 10, name_y - conf_size[1] - 5),
                            (x1 + text_size[0] + conf_size[0] + 15, name_y + 5),
                            (50, 50, 100), -1)
                cv2.putText(frame, confidence_text, 
                           (x1 + text_size[0] + 12, name_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 200), 1)
    
    # === BOTTOM-CENTER: Status (clean, minimal) ===
    status_y = h - 80
    status_bg_height = 50
    status_bg_width = 250
    
    # Position at bottom-center
    bg_x = w // 2 - status_bg_width // 2
    bg_y = status_y
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x, bg_y), 
                 (bg_x + status_bg_width, bg_y + status_bg_height), 
                 (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw status border
    cv2.rectangle(frame, (bg_x, bg_y), 
                 (bg_x + status_bg_width, bg_y + status_bg_height), 
                 status_color, 2)
    
    # Add status text
    status_text = f"STATUS: {status}"
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = bg_x + (status_bg_width - text_size[0]) // 2
    text_y = bg_y + 35
    
    cv2.putText(frame, status_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # === MINIMAL BOTTOM-LEFT INFO ===
    if face_detected:
        # Small blink indicator
        if blink_detected:
            cv2.putText(frame, "BLINK", (20, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # === NO DRIVER DETECTED MESSAGE ===
    if not face_detected:
        # Center the message
        message = "NO DRIVER DETECTED"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x-20, text_y-40), 
                     (text_x+text_size[0]+20, text_y+20), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text
        cv2.putText(frame, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        
        sub_message = "Position face in camera view"
        sub_size = cv2.getTextSize(sub_message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        sub_x = (w - sub_size[0]) // 2
        
        cv2.putText(frame, sub_message, (sub_x, text_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame

def create_dashboard2(dashboard_data, ear_history, mar_history, ml_model):
    """Create Dashboard 2: Analytics Dashboard"""
    # Create black canvas
    dashboard = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Header
    cv2.putText(dashboard, "DASHBOARD 2: ANALYTICS", (20, 40),
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Left Column: Status
    cv2.rectangle(dashboard, (20, 100), (380, 250), (40, 40, 40), -1)
    cv2.rectangle(dashboard, (20, 100), (380, 250), (0, 200, 200), 2)
    
    cv2.putText(dashboard, "SYSTEM STATUS", (30, 130),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Get current driver info from classifier
    driver_name = driver_classifier.current_driver_name
    confidence = driver_classifier.current_confidence
    
    # Determine driver status
    if driver_name == "Unknown (Train Model)":
        driver_status = "Driver: Unknown (Training Needed)"
        confidence_status = "Run: python train_driver_classifier.py"
    elif driver_name != "Unknown" and confidence > 0.5:
        driver_status = f"Driver: {driver_name}"
        confidence_status = f"Confidence: {confidence:.1%}"
    elif dashboard_data['face_detected']:
        driver_status = "Driver: Detected (Unknown)"
        confidence_status = "Training needed"
    else:
        driver_status = "Driver: Not Detected"
        confidence_status = ""
    
    status_items = [
        f"Status: {dashboard_data['status']}",
        driver_status,
        confidence_status,
        f"FPS: {dashboard_data['fps']:.1f}",
        f"Time: {dashboard_data['timestamp']}",
        f"Total Blinks: {dashboard_data['blinks']}",
        f"ML Model: {'Active' if dashboard_data['ml_enabled'] else 'Inactive'}"
    ]
    
    y_offset = 160
    for item in status_items:
        if item:  # Skip empty items
            color = (200, 200, 200)
            if "Training Needed" in item or "Training needed" in item:
                color = (0, 165, 255)  # Orange for warnings
            cv2.putText(dashboard, item, (30, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    # Right Column: Metrics
    cv2.rectangle(dashboard, (420, 100), (780, 250), (40, 40, 40), -1)
    cv2.rectangle(dashboard, (420, 100), (780, 250), (0, 200, 200), 2)
    
    cv2.putText(dashboard, "CURRENT METRICS", (430, 130),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    metrics = [
        f"EAR: {dashboard_data['ear']:.3f}",
        f"MAR: {dashboard_data['mar']:.3f}",
        f"ML Accuracy: {dashboard_data['ml_accuracy']:.1%}",
        f"ML Confidence: {dashboard_data['ml_confidence']:.1%}"
    ]
    
    for i, metric in enumerate(metrics):
        cv2.putText(dashboard, metric, (430, 160 + i*25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Graph area - EAR History
    if len(ear_history) > 0:
        graph_y = 280
        graph_height = 150
        graph_width = 760
        
        # Draw graph background
        cv2.rectangle(dashboard, (20, graph_y), (780, graph_y + graph_height), (30, 30, 30), -1)
        cv2.rectangle(dashboard, (20, graph_y), (780, graph_y + graph_height), (0, 200, 200), 2)
        
        cv2.putText(dashboard, "EAR HISTORY (Last 100 frames)", (30, graph_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Plot EAR values
        max_ear = max(ear_history) if ear_history else 0.4
        min_ear = min(ear_history) if ear_history else 0.1
        
        for i in range(1, len(ear_history)):
            x1 = 20 + int((i-1) * graph_width / len(ear_history))
            x2 = 20 + int(i * graph_width / len(ear_history))
            
            if max_ear > min_ear:
                y1 = graph_y + graph_height - int((ear_history[i-1] - min_ear) * graph_height / (max_ear - min_ear))
                y2 = graph_y + graph_height - int((ear_history[i] - min_ear) * graph_height / (max_ear - min_ear))
                cv2.line(dashboard, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # ML Model Info
    if ml_model.is_available:
        info_y = 450
        cv2.rectangle(dashboard, (20, info_y), (780, 580), (40, 40, 40), -1)
        cv2.rectangle(dashboard, (20, info_y), (780, 580), (0, 200, 200), 2)
        
        cv2.putText(dashboard, "ML MODEL INFORMATION", (30, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        info_items = [
            f"Model Type: Random Forest",
            f"Training Accuracy: {ml_model.accuracy:.1%}",
            f"Number of Features: {ml_model.num_features}",
            f"Current Confidence: {dashboard_data['ml_confidence']:.1%}"
        ]
        
        for i, item in enumerate(info_items):
            cv2.putText(dashboard, item, (30, info_y + 55 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Add Driver Recognition Status
    if hasattr(driver_classifier, 'model_loaded'):
        status_y = 580
        status_color = (0, 255, 0) if driver_classifier.model_loaded else (0, 165, 255)
        status_text = "Driver Recognition: ACTIVE" if driver_classifier.model_loaded else "Driver Recognition: INACTIVE"
        cv2.putText(dashboard, status_text, (20, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    return dashboard

def train_driver_classifier():
    """Train the driver classifier from database"""
    print("\n" + "="*60)
    print("🤖 TRAINING DRIVER CLASSIFIER")
    print("="*60)
    
    if driver_classifier.model_loaded:
        print("✅ Model already loaded and ready!")
        return True
    
    print("⚠️ Model not loaded. You need to train the classifier.")
    print("\nTo train the classifier:")
    print("1. Collect driver face images using the registration system")
    print("2. Run: python train_driver_classifier.py")
    print("3. Make sure you have driver faces in your database")
    
    return False