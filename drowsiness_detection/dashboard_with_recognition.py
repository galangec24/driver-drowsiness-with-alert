"""
Dashboard visualization functions with integrated driver recognition
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

class RealTimeDriverClassifier:
    def __init__(self, model_path='driver_classifier.pkl', db_path='../backend/drivers.db'):
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
        
        # Load or create model
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.label_encoder = joblib.load('label_encoder.pkl')
                self.model_loaded = True
                print("✅ Driver classifier model loaded")
            except:
                self.model = SVC(kernel='linear', probability=True)
                self.label_encoder = LabelEncoder()
                self.model_loaded = False
        else:
            self.model = SVC(kernel='linear', probability=True)
            self.label_encoder = LabelEncoder()
            self.model_loaded = False
        
        # Recognition tracking
        self.last_recognition_time = 0
        self.recognition_interval = 5  # Recognize every 5 seconds
        self.current_driver_name = "Unknown"
        self.current_driver_id = None
        self.current_confidence = 0.0
    
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
            
            print(f"✅ Loaded {len(self.driver_names)} driver names from database")
        except Exception as e:
            print(f"⚠️  Error loading driver names: {e}")
            # Create dummy data for testing
            self.driver_names = {
                'D001': 'John Doe',
                'D002': 'Jane Smith',
                'D003': 'Robert Johnson',
                'D004': 'Maria Garcia'
            }
    
    def extract_face_features(self, frame):
        """Extract facial features from video frame"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                features = []
                
                # Extract key landmarks only for faster processing
                key_indices = [33, 133, 362, 263, 1, 4, 13, 14, 78, 308]
                
                for idx in key_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        features.extend([landmark.x, landmark.y])
                
                return np.array(features)
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return None
    
    def recognize_driver(self, frame):
        """Recognize driver from video frame"""
        if not self.model_loaded:
            return "Unknown", 0.0
        
        current_time = time.time()
        
        # Only recognize at intervals to save CPU
        if current_time - self.last_recognition_time < self.recognition_interval:
            return self.current_driver_name, self.current_confidence
        
        features = self.extract_face_features(frame)
        if features is None:
            self.current_driver_name = "Unknown"
            self.current_confidence = 0.0
            return self.current_driver_name, self.current_confidence
        
        try:
            # Reshape for prediction
            features = features.reshape(1, -1)
            
            # Predict
            probabilities = self.model.predict_proba(features)[0]
            max_prob = np.max(probabilities)
            pred_index = np.argmax(probabilities)
            
            if max_prob > 0.6:  # Confidence threshold
                driver_id = self.label_encoder.inverse_transform([pred_index])[0]
                self.current_driver_id = driver_id
                self.current_driver_name = self.driver_names.get(driver_id, f"Driver_{driver_id}")
                self.current_confidence = float(max_prob)
                print(f"👤 Recognized: {self.current_driver_name} ({self.current_confidence:.1%})")
            else:
                self.current_driver_name = "Unknown"
                self.current_driver_id = None
                self.current_confidence = float(max_prob)
            
            self.last_recognition_time = current_time
            return self.current_driver_name, self.current_confidence
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.0

# Initialize global classifier
driver_classifier = RealTimeDriverClassifier()

def create_dashboard1(frame, ear, mar, status, blink_detected, face_detected,
                     fps, total_blinks, blink_rate, ml_model, face_bbox=None):
    """Create Dashboard 1: Live Monitoring with Driver Recognition"""
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
    
    # === TOP-RIGHT: FPS only ===
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # === DRIVER RECOGNITION ===
    if face_detected and face_bbox:
        x1, y1, x2, y2 = face_bbox
        
        # Draw face bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Get driver recognition
        driver_name, confidence = driver_classifier.recognize_driver(frame)
        
        # Show driver info above face box
        name_y = max(30, y1 - 10)
        
        if driver_name != "Unknown" and confidence > 0.6:
            # Recognized driver
            name_text = f"{driver_name}"
            conf_text = f"{confidence:.0%}"
            
            # Draw name background
            name_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, 
                         (x1 - 5, name_y - name_size[1] - 5),
                         (x1 + name_size[0] + 5, name_y + 5),
                         (0, 100, 0), -1)
            
            # Draw name
            cv2.putText(frame, name_text, (x1, name_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw confidence to the right
            cv2.putText(frame, conf_text, 
                       (x1 + name_size[0] + 10, name_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
        else:
            # Unknown driver
            unknown_text = "Unknown Driver"
            text_size = cv2.getTextSize(unknown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            
            cv2.rectangle(frame, 
                         (x1 - 5, name_y - text_size[1] - 5),
                         (x1 + text_size[0] + 5, name_y + 5),
                         (50, 50, 50), -1)
            
            cv2.putText(frame, unknown_text, (x1, name_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # === BOTTOM-CENTER: Status ===
    status_y = h - 80
    status_bg_width = 250
    status_bg_height = 50
    
    bg_x = w // 2 - status_bg_width // 2
    bg_y = status_y
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x, bg_y), 
                 (bg_x + status_bg_width, bg_y + status_bg_height), 
                 (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Border
    cv2.rectangle(frame, (bg_x, bg_y), 
                 (bg_x + status_bg_width, bg_y + status_bg_height), 
                 status_color, 2)
    
    # Text
    status_text = f"STATUS: {status}"
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = bg_x + (status_bg_width - text_size[0]) // 2
    text_y = bg_y + 35
    
    cv2.putText(frame, status_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # === MINIMAL INFO ===
    if face_detected and blink_detected:
        cv2.putText(frame, "BLINK", (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # === NO DRIVER MESSAGE ===
    if not face_detected:
        message = "NO DRIVER DETECTED"
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x-20, text_y-40), 
                     (text_x+text_size[0]+20, text_y+20), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, message, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    return frame

def create_dashboard2(dashboard_data, ear_history, mar_history, ml_model):
    """Create Dashboard 2: Analytics Dashboard"""
    dashboard = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Header
    cv2.putText(dashboard, "DASHBOARD 2: ANALYTICS", (20, 40),
               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    # Left: Status
    cv2.rectangle(dashboard, (20, 100), (380, 250), (40, 40, 40), -1)
    cv2.rectangle(dashboard, (20, 100), (380, 250), (0, 200, 200), 2)
    
    cv2.putText(dashboard, "SYSTEM STATUS", (30, 130),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Get driver info
    driver_name = driver_classifier.current_driver_name
    driver_conf = driver_classifier.current_confidence
    
    if driver_name != "Unknown" and driver_conf > 0.6:
        driver_status = f"Driver: {driver_name}"
        conf_status = f"Confidence: {driver_conf:.1%}"
    elif dashboard_data['face_detected']:
        driver_status = "Driver: Unknown"
        conf_status = "Confidence: Low"
    else:
        driver_status = "Driver: Not Detected"
        conf_status = ""
    
    status_items = [
        f"Status: {dashboard_data['status']}",
        driver_status,
        conf_status,
        f"FPS: {dashboard_data['fps']:.1f}",
        f"Time: {dashboard_data['timestamp']}",
        f"Total Blinks: {dashboard_data['blinks']}",
        f"ML Model: {'Active' if dashboard_data['ml_enabled'] else 'Inactive'}"
    ]
    
    y = 160
    for item in status_items:
        if item:
            cv2.putText(dashboard, item, (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y += 20
    
    # Right: Metrics
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
    
    # EAR History
    if ear_history:
        graph_y = 280
        graph_h = 150
        graph_w = 760
        
        cv2.rectangle(dashboard, (20, graph_y), (780, graph_y + graph_h), (30, 30, 30), -1)
        cv2.rectangle(dashboard, (20, graph_y), (780, graph_y + graph_h), (0, 200, 200), 2)
        
        cv2.putText(dashboard, "EAR HISTORY (Last 100 frames)", (30, graph_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        max_ear = max(ear_history)
        min_ear = min(ear_history)
        
        for i in range(1, len(ear_history)):
            x1 = 20 + int((i-1) * graph_w / len(ear_history))
            x2 = 20 + int(i * graph_w / len(ear_history))
            
            if max_ear > min_ear:
                y1 = graph_y + graph_h - int((ear_history[i-1] - min_ear) * graph_h / (max_ear - min_ear))
                y2 = graph_y + graph_h - int((ear_history[i] - min_ear) * graph_h / (max_ear - min_ear))
                cv2.line(dashboard, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # ML Info
    if ml_model['is_available']:
        info_y = 450
        cv2.rectangle(dashboard, (20, info_y), (780, 580), (40, 40, 40), -1)
        cv2.rectangle(dashboard, (20, info_y), (780, 580), (0, 200, 200), 2)
        
        cv2.putText(dashboard, "ML MODEL INFORMATION", (30, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        info_items = [
            f"Model Type: Random Forest",
            f"Training Accuracy: {ml_model['accuracy']:.1%}",
            f"Number of Features: {ml_model['num_features']}",
            f"Current Confidence: {dashboard_data['ml_confidence']:.1%}"
        ]
        
        for i, item in enumerate(info_items):
            cv2.putText(dashboard, item, (30, info_y + 55 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return dashboard

def train_driver_classifier():
    """Train driver classifier"""
    return driver_classifier.model_loaded