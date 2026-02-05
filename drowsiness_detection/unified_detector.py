"""
ENHANCED DRIVER DROWSINESS DETECTION SYSTEM
With Transfer Learning model, facial landmark analysis, and real-time alerts
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import json
import joblib
import mediapipe as mp
import time
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚗 ENHANCED DRIVER DROWSINESS DETECTION SYSTEM")
print("="*80)

class DriverDrowsinessSystem:
    def __init__(self):
        # Get correct paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.models_dir = os.path.join(project_root, "models")
        
        print(f"📁 Project root: {project_root}")
        print(f"📁 Models directory: {self.models_dir}")
        
        # Debug mode
        self.debug_mode = False
        self.visual_debug = False
        
        # Camera selection
        self.camera_index = self.select_camera()
        
        print("\n🔍 Loading models...")
        
        # Load driver recognition model
        try:
            mapping_path = os.path.join(self.models_dir, "driver_mapping.json")
            if not os.path.exists(mapping_path):
                raise FileNotFoundError(f"File not found: {mapping_path}")
            
            with open(mapping_path, 'r') as f:
                self.driver_mapping = json.load(f)
            
            self.driver_names = self.driver_mapping['driver_names']
            
            # Load model components
            self.driver_model = joblib.load(os.path.join(self.models_dir, "driver_model.pkl"))
            self.driver_encoder = joblib.load(os.path.join(self.models_dir, "driver_encoder.pkl"))
            self.driver_scaler = joblib.load(os.path.join(self.models_dir, "driver_scaler.pkl"))
            
            print(f"✅ Driver model loaded")
            print(f"   Registered drivers: {list(self.driver_names.values())}")
            print(f"   Training accuracy: {self.driver_mapping.get('accuracy', 0):.1%}")
            
        except Exception as e:
            print(f"⚠️ Driver model error: {e}")
            self.driver_model = None
            self.driver_names = {}
            print("💡 Driver recognition disabled")
        
        # Load ENHANCED drowsiness model (Transfer Learning)
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            drowsiness_model_path = os.path.join(self.models_dir, "drowsiness_model.h5")
            if not os.path.exists(drowsiness_model_path):
                # Try enhanced model
                drowsiness_model_path = os.path.join(self.models_dir, "drowsiness_model_enhanced.h5")
                if not os.path.exists(drowsiness_model_path):
                    raise FileNotFoundError("No drowsiness model found")
            
            self.drowsiness_model = tf.keras.models.load_model(
                drowsiness_model_path,
                compile=False
            )
            
            # Try to load class info
            drowsiness_classes_path = os.path.join(self.models_dir, "drowsiness_classes.json")
            if not os.path.exists(drowsiness_classes_path):
                drowsiness_classes_path = os.path.join(self.models_dir, "drowsiness_classes_enhanced.json")
            
            if os.path.exists(drowsiness_classes_path):
                with open(drowsiness_classes_path, 'r') as f:
                    drowsiness_info = json.load(f)
                
                # Get class mapping
                class_indices = drowsiness_info['class_indices']
                self.drowsiness_classes = {v: k for k, v in class_indices.items()}
                
                # Get input shape from model info or from model
                if 'input_shape' in drowsiness_info:
                    self.drowsiness_input_shape = drowsiness_info['input_shape']
                else:
                    # Infer from model
                    self.drowsiness_input_shape = self.drowsiness_model.input_shape[1:4]
                
                print(f"✅ Enhanced drowsiness model loaded")
                print(f"   Model: {os.path.basename(drowsiness_model_path)}")
                print(f"   Input shape: {self.drowsiness_input_shape}")
                print(f"   Classes: {list(class_indices.keys())}")
                
                if 'performance' in drowsiness_info:
                    perf = drowsiness_info['performance']
                    if 'val_accuracy' in perf:
                        print(f"   Model accuracy: {perf['val_accuracy']:.1%}")
                
            else:
                # Default mapping
                self.drowsiness_classes = {0: "Drowsy", 1: "Non_Drowsy", 2: "Yawning"}
                self.drowsiness_input_shape = [224, 224, 3]  # Default for MobileNetV2
                print("⚠️ Using default class mapping")
            
            # Confidence threshold
            self.drowsiness_threshold = 0.65
            print(f"   Confidence threshold: {self.drowsiness_threshold}")
            
        except Exception as e:
            print(f"❌ Drowsiness model error: {e}")
            self.drowsiness_model = None
            self.drowsiness_classes = {0: "Drowsy", 1: "Non_Drowsy", 2: "Yawning"}
            self.drowsiness_input_shape = [128, 128, 3]
            print("💡 Using fallback detection method")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7  # Higher confidence for better accuracy
        )
        
        # Initialize MediaPipe Face Mesh for enhanced landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize MediaPipe Holistic for full pose estimation
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Tracking variables
        self.last_face_box = None
        self.face_lost_count = 0
        self.max_face_lost = 15
        
        # State tracking for smoothing
        self.driver_history = []
        self.drowsiness_history = []
        self.history_size = 9  # Increased for better smoothing
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Enhanced alert system
        self.alert_active = False
        self.alert_start_time = 0
        self.consecutive_drowsy_frames = 0
        self.consecutive_yawning_frames = 0
        self.drowsy_threshold = 8  # Faster response
        self.yawning_threshold = 5
        self.alert_cooldown = 5.0  # seconds
        
        # Eye/Mouth analysis parameters
        self.eye_closed_threshold = 0.22  # EAR below this means eyes are closed
        self.eye_blink_threshold = 0.25   # EAR for blink detection
        self.yawning_threshold_mar = 0.65  # MAR above this means yawning
        
        # Blink detection
        self.blink_counter = 0
        self.blink_start_time = 0
        self.blink_duration = 0
        self.eye_closed_frames = 0
        self.blink_history = []
        self.max_blink_history = 30
        
        # PERCLOS (Percentage of Eyelid Closure) calculation
        self.perclos_window = 60  # 60 frames for 1 minute at 1 FPS
        self.eye_state_history = []
        
        # Performance metrics
        self.detection_stats = {
            'total_frames': 0,
            'drowsy_frames': 0,
            'yawning_frames': 0,
            'blinks': 0,
            'alerts_triggered': 0
        }
        
        # Face quality metrics
        self.face_quality_threshold = 0.6
        
        print(f"\n📊 System initialized with camera index: {self.camera_index}")
        print("✅ Enhanced system ready for detection")
    
    def select_camera(self):
        """Try to find and use external camera first"""
        print("\n📷 Scanning for cameras...")
        
        # Try external cameras first (indices 1, 2, 3...)
        for camera_index in [1, 2, 3, 0]:  # Try external first, then built-in
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    if camera_index == 0:
                        print(f"   Using built-in camera (index {camera_index})")
                    else:
                        print(f"   ✅ Found external camera (index {camera_index})")
                    return camera_index
        
        print("   ⚠️ No cameras found, using default index 0")
        return 0
    
    def extract_face_features(self, bbox, frame):
        """Extract features from face bounding box for driver recognition"""
        h, w = frame.shape[:2]
        
        features = [
            bbox.xmin,
            bbox.ymin,
            bbox.width,
            bbox.height,
            
            # Absolute coordinates
            bbox.xmin * w,
            bbox.ymin * h,
            bbox.width * w,
            bbox.height * h,
            
            # Center coordinates
            bbox.xmin + bbox.width/2,
            bbox.ymin + bbox.height/2,
            
            # Aspect ratios
            w / max(h, 1),
            bbox.width / max(bbox.height, 0.001),
            
            # Additional features
            bbox.xmin * bbox.ymin,
            bbox.width * bbox.height,
            (bbox.xmin + bbox.width/2) / max(w, 1),
            (bbox.ymin + bbox.height/2) / max(h, 1)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def recognize_driver(self, features):
        """Recognize driver from face features"""
        if self.driver_model is None or not self.driver_names:
            return "Unknown", 0.5
        
        try:
            features = features.reshape(1, -1)
            
            # Match feature dimensions
            if hasattr(self.driver_scaler, 'mean_'):
                n_expected = self.driver_scaler.mean_.shape[0]
                if features.shape[1] > n_expected:
                    features = features[:, :n_expected]
                elif features.shape[1] < n_expected:
                    padding = np.zeros((1, n_expected - features.shape[1]))
                    features = np.hstack([features, padding])
            
            # Scale features
            features_scaled = self.driver_scaler.transform(features)
            
            # Predict
            if hasattr(self.driver_model, 'predict_proba'):
                proba = self.driver_model.predict_proba(features_scaled)[0]
                pred_idx = np.argmax(proba)
                confidence = proba[pred_idx]
            else:
                pred_idx = self.driver_model.predict(features_scaled)[0]
                confidence = 0.8
            
            # Get driver name
            driver_id = self.driver_encoder.inverse_transform([pred_idx])[0]
            driver_id_str = str(driver_id)
            driver_name = self.driver_names.get(driver_id_str, f"Driver_{pred_idx}")
            
            return driver_name, confidence
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Driver recognition error: {e}")
            return "Unknown", 0.0
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Get the required landmarks
        p1 = np.array([landmarks.landmark[eye_indices[0]].x, landmarks.landmark[eye_indices[0]].y])
        p2 = np.array([landmarks.landmark[eye_indices[1]].x, landmarks.landmark[eye_indices[1]].y])
        p3 = np.array([landmarks.landmark[eye_indices[2]].x, landmarks.landmark[eye_indices[2]].y])
        p4 = np.array([landmarks.landmark[eye_indices[3]].x, landmarks.landmark[eye_indices[3]].y])
        p5 = np.array([landmarks.landmark[eye_indices[4]].x, landmarks.landmark[eye_indices[4]].y])
        p6 = np.array([landmarks.landmark[eye_indices[5]].x, landmarks.landmark[eye_indices[5]].y])
        
        # Calculate vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Calculate horizontal distance
        h = np.linalg.norm(p1 - p4)
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def calculate_mouth_aspect_ratio(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR)"""
        # Mouth landmarks
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]
        
        # Vertical points (top and bottom of mouth)
        p61 = np.array([landmarks.landmark[61].x, landmarks.landmark[61].y])
        p146 = np.array([landmarks.landmark[146].x, landmarks.landmark[146].y])
        
        # Horizontal points (left and right corners)
        p91 = np.array([landmarks.landmark[91].x, landmarks.landmark[91].y])
        p181 = np.array([landmarks.landmark[181].x, landmarks.landmark[181].y])
        
        # Calculate distances
        vertical = np.linalg.norm(p61 - p146)
        horizontal = np.linalg.norm(p91 - p181)
        
        # MAR formula
        mar = vertical / (horizontal + 1e-6)
        return mar
    
    def analyze_facial_landmarks(self, face_roi):
        """Enhanced facial landmark analysis with eye and mouth tracking"""
        try:
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Eye landmarks indices (MediaPipe 468 landmarks)
                left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173, 246, 161]
                right_eye_indices = [362, 263, 386, 387, 388, 389, 390, 466, 466, 397]
                
                # Calculate EAR for both eyes
                left_ear = self.calculate_eye_aspect_ratio(landmarks, left_eye_indices[:6])
                right_ear = self.calculate_eye_aspect_ratio(landmarks, right_eye_indices[:6])
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Calculate MAR
                mar = self.calculate_mouth_aspect_ratio(landmarks)
                
                # Blink detection
                is_eye_closed = avg_ear < self.eye_closed_threshold
                is_blinking = avg_ear < self.eye_blink_threshold
                
                # Update blink counter
                if is_blinking and self.blink_start_time == 0:
                    self.blink_start_time = time.time()
                elif not is_blinking and self.blink_start_time > 0:
                    self.blink_duration = time.time() - self.blink_start_time
                    if 0.1 < self.blink_duration < 0.4:  # Normal blink duration
                        self.blink_counter += 1
                        self.detection_stats['blinks'] += 1
                    self.blink_start_time = 0
                
                # Update eye state history for PERCLOS
                self.eye_state_history.append(1 if is_eye_closed else 0)
                if len(self.eye_state_history) > self.perclos_window:
                    self.eye_state_history.pop(0)
                
                # Calculate PERCLOS
                if len(self.eye_state_history) > 0:
                    perclos = sum(self.eye_state_history) / len(self.eye_state_history)
                else:
                    perclos = 0
                
                # Visual debug landmarks
                if self.visual_debug:
                    h, w = face_roi.shape[:2]
                    debug_img = face_roi.copy()
                    
                    # Draw eye landmarks
                    for idx in left_eye_indices[:6]:
                        pt = landmarks.landmark[idx]
                        x, y = int(pt.x * w), int(pt.y * h)
                        cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
                    
                    for idx in right_eye_indices[:6]:
                        pt = landmarks.landmark[idx]
                        x, y = int(pt.x * w), int(pt.y * h)
                        cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)
                    
                    # Draw mouth landmarks
                    mouth_indices = [61, 146, 91, 181]
                    for idx in mouth_indices:
                        pt = landmarks.landmark[idx]
                        x, y = int(pt.x * w), int(pt.y * h)
                        cv2.circle(debug_img, (x, y), 2, (255, 0, 0), -1)
                    
                    # Show EAR and MAR values
                    cv2.putText(debug_img, f"EAR: {avg_ear:.3f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(debug_img, f"MAR: {mar:.3f}", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(debug_img, f"PERCLOS: {perclos:.1%}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    cv2.imshow("Facial Landmarks Debug", debug_img)
                
                # Determine state based on thresholds
                if avg_ear < self.eye_closed_threshold:
                    if perclos > 0.3:  # High PERCLOS indicates drowsiness
                        return "Drowsy", 0.9, avg_ear, mar, perclos
                    else:
                        return "Drowsy", 0.85, avg_ear, mar, perclos
                elif mar > self.yawning_threshold_mar:
                    return "Yawning", 0.88, avg_ear, mar, perclos
                else:
                    return "Non_Drowsy", 0.95, avg_ear, mar, perclos
                    
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Landmark analysis error: {e}")
        
        return None, 0.0, 0.0, 0.0, 0.0
    
    def detect_drowsiness_cnn(self, face_roi):
        """Detect drowsiness using CNN model"""
        if self.drowsiness_model is None:
            return self.fallback_detection(face_roi)
        
        try:
            # Check face size
            if face_roi.shape[0] < 60 or face_roi.shape[1] < 60:
                return "Non_Drowsy", 0.5
            
            # Preprocess image according to model's expected input shape
            target_size = (self.drowsiness_input_shape[0], self.drowsiness_input_shape[1])
            img = cv2.resize(face_roi, target_size)
            
            # Ensure 3 channels
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            # Convert to RGB (required for models trained on RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize
            img = img.astype('float32') / 255.0
            
            # Apply slight enhancement
            img = cv2.normalize(img, None, alpha=0, beta=1, 
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            img = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.drowsiness_model.predict(img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            # Get class name
            if class_idx in self.drowsiness_classes:
                state = self.drowsiness_classes[class_idx]
            else:
                # If mapping is wrong, use index-based fallback
                states = ["Drowsy", "Non_Drowsy", "Yawning"]
                state = states[min(class_idx, len(states)-1)]
            
            # Debug: Print raw predictions
            if self.debug_mode:
                preds_str = " | ".join([f"{self.drowsiness_classes.get(i, f'Class_{i}')}: {p:.2%}" 
                                      for i, p in enumerate(predictions) if i in self.drowsiness_classes])
                print(f"🔍 CNN predictions: [{preds_str}]")
                print(f"🔍 Predicted: {state} (idx={class_idx}, conf={confidence:.2%})")
            
            # Apply confidence threshold
            if confidence < self.drowsiness_threshold:
                if self.debug_mode and confidence > 0.3:
                    print(f"⚠️ Below threshold: {confidence:.2%} < {self.drowsiness_threshold:.2%}")
                return "Non_Drowsy", confidence
            
            return state, confidence
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ CNN detection error: {e}")
            return self.fallback_detection(face_roi)
    
    def fallback_detection(self, face_roi):
        """Fallback detection using simple rules"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            gray_eq = cv2.equalizeHist(gray)
            
            # Edge detection
            edges = cv2.Canny(gray_eq, 50, 150)
            edge_density = np.sum(edges > 0) / (face_roi.shape[0] * face_roi.shape[1])
            
            # Calculate brightness variance
            brightness_var = np.var(gray_eq)
            
            if self.debug_mode:
                print(f"🔍 Fallback - Edge density: {edge_density:.3f}, Brightness var: {brightness_var:.1f}")
            
            # Improved rule-based detection
            if edge_density < 0.005 and brightness_var < 500:
                return "Drowsy", 0.7
            elif edge_density > 0.03:  # High edge density might indicate open eyes
                return "Non_Drowsy", 0.8
            else:
                return "Non_Drowsy", 0.7
                
        except:
            return "Non_Drowsy", 0.5
    
    def detect_drowsiness(self, face_roi):
        """Main drowsiness detection combining multiple methods"""
        # Method 1: Facial landmark analysis (most reliable)
        landmark_state, landmark_conf, ear, mar, perclos = self.analyze_facial_landmarks(face_roi)
        
        # If landmark detection worked and found drowsy/yawn with high confidence, use it
        if landmark_state and landmark_state != "Non_Drowsy" and landmark_conf > 0.8:
            if self.debug_mode:
                print(f"✅ Using landmark detection: {landmark_state} (EAR: {ear:.3f}, MAR: {mar:.3f}, PERCLOS: {perclos:.1%})")
            return landmark_state, landmark_conf, ear, mar, perclos
        
        # Method 2: CNN model
        cnn_state, cnn_conf = self.detect_drowsiness_cnn(face_roi)
        
        # If CNN found drowsy/yawn with good confidence, use it
        if cnn_state != "Non_Drowsy" and cnn_conf > 0.75:
            if self.debug_mode:
                print(f"✅ Using CNN detection: {cnn_state} ({cnn_conf:.1%})")
            return cnn_state, cnn_conf, ear, mar, perclos
        
        # Method 3: If we have landmark results, use them
        if landmark_state:
            if self.debug_mode:
                print(f"✅ Using landmark detection (normal): {landmark_state}")
            return landmark_state, landmark_conf, ear, mar, perclos
        
        # Method 4: Use CNN results
        if self.debug_mode:
            print(f"✅ Using CNN detection: {cnn_state}")
        return cnn_state, cnn_conf, ear, mar, perclos
    
    def assess_face_quality(self, face_roi, bbox):
        """Assess quality of face detection"""
        try:
            h, w = face_roi.shape[:2]
            
            # Check size
            if h < 50 or w < 50:
                return 0.0, "Face too small"
            
            # Check aspect ratio
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:
                return 0.4, f"Bad aspect ratio: {aspect_ratio:.2f}"
            
            # Check brightness
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness_mean = np.mean(gray)
            
            if brightness_mean < 50:
                return 0.3, f"Too dark: {brightness_mean:.0f}"
            elif brightness_mean > 200:
                return 0.3, f"Too bright: {brightness_mean:.0f}"
            
            # Check blurriness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:
                return 0.5, f"Blurry: {laplacian_var:.0f}"
            
            # Calculate quality score
            size_score = min(1.0, h / 200.0)
            brightness_score = 1.0 - abs(brightness_mean - 127) / 127
            sharpness_score = min(1.0, laplacian_var / 200.0)
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 0.3
            
            quality_score = (size_score + brightness_score + sharpness_score + aspect_score) / 4.0
            
            if quality_score < self.face_quality_threshold:
                return quality_score, f"Low quality: {quality_score:.2f}"
            
            return quality_score, "Good"
            
        except Exception as e:
            return 0.0, f"Error: {str(e)}"
    
    def update_history(self, driver_name, driver_conf, drowsiness_state, drowsiness_conf, ear, mar, perclos):
        """Update history for smoothing results"""
        self.driver_history.append((driver_name, driver_conf))
        self.drowsiness_history.append((drowsiness_state, drowsiness_conf, ear, mar, perclos))
        
        if len(self.driver_history) > self.history_size:
            self.driver_history.pop(0)
            self.drowsiness_history.pop(0)
    
    def get_smoothed_results(self):
        """Get smoothed results from history"""
        if not self.drowsiness_history:
            return "Unknown", 0.0, "Non_Drowsy", 0.8, 0.0, 0.0, 0.0
        
        # Smooth driver recognition
        driver_scores = {}
        for name, conf in self.driver_history:
            driver_scores[name] = driver_scores.get(name, 0) + conf
        
        if driver_scores:
            best_driver = max(driver_scores, key=driver_scores.get)
            avg_driver_conf = np.mean([conf for name, conf in self.driver_history 
                                      if name == best_driver])
        else:
            best_driver = "Unknown"
            avg_driver_conf = 0.5
        
        # Smooth drowsiness detection
        state_scores = {}
        ear_values = []
        mar_values = []
        perclos_values = []
        
        for state, conf, ear, mar, perclos in self.drowsiness_history:
            state_scores[state] = state_scores.get(state, 0) + conf
            ear_values.append(ear)
            mar_values.append(mar)
            perclos_values.append(perclos)
        
        if state_scores:
            best_state = max(state_scores, key=state_scores.get)
            
            # Get confidence for best state
            best_state_confs = [conf for state, conf, ear, mar, perclos in self.drowsiness_history 
                               if state == best_state]
            avg_state_conf = np.mean(best_state_confs) if best_state_confs else 0.7
            
            # Get average biometric values
            avg_ear = np.mean(ear_values) if ear_values else 0.0
            avg_mar = np.mean(mar_values) if mar_values else 0.0
            avg_perclos = np.mean(perclos_values) if perclos_values else 0.0
            
            # Enhanced consistency check
            if best_state != "Non_Drowsy":
                state_count = sum(1 for state, _, _, _, _ in self.drowsiness_history if state == best_state)
                total_frames = len(self.drowsiness_history)
                
                # Require strong consistency for drowsy states
                if best_state == "Drowsy":
                    required_ratio = 0.7  # 70% of recent frames must be drowsy
                else:  # Yawning
                    required_ratio = 0.6  # 60% of recent frames must be yawning
                
                actual_ratio = state_count / total_frames
                
                if actual_ratio < required_ratio:
                    if self.debug_mode:
                        print(f"⚠️ Not consistent enough for {best_state}: {state_count}/{total_frames} ({actual_ratio:.1%} < {required_ratio:.0%})")
                    best_state = "Non_Drowsy"
                    avg_state_conf = 0.8
        else:
            best_state = "Non_Drowsy"
            avg_state_conf = 0.8
            avg_ear = 0.0
            avg_mar = 0.0
            avg_perclos = 0.0
        
        if self.debug_mode and len(self.drowsiness_history) >= 3:
            recent_states = [state for state, _, _, _, _ in self.drowsiness_history[-3:]]
            print(f"📊 Recent states: {recent_states} → Smoothed: {best_state}")
        
        return best_driver, avg_driver_conf, best_state, avg_state_conf, avg_ear, avg_mar, avg_perclos
    
    def draw_enhanced_info(self, frame, x, y, w, h, driver_name, driver_conf, 
                          drowsiness_state, drowsiness_conf, ear, mar, perclos, fps, face_quality):
        """Draw enhanced detection information on frame"""
        # Determine colors based on state
        if drowsiness_state == "Drowsy":
            box_color = (0, 0, 255)  # Red
            status_color = (0, 0, 255)
            status_text = "🚨 DROWSY"
            alert_level = "HIGH"
        elif drowsiness_state == "Yawning":
            box_color = (0, 165, 255)  # Orange
            status_color = (0, 165, 255)
            status_text = "⚠️ YAWNING"
            alert_level = "MEDIUM"
        else:  # Non_Drowsy
            box_color = (0, 255, 0)  # Green
            status_color = (0, 255, 0)
            status_text = "✅ ALERT"
            alert_level = "LOW"
        
        # Draw enhanced face bounding box with shadow effect
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)
        cv2.rectangle(frame, (x+2, y+2), (x + w - 2, y + h - 2), (255, 255, 255), 1)
        
        # Draw status box at top with gradient effect
        status_box_height = 130
        cv2.rectangle(frame, (x, y - status_box_height), 
                     (x + w, y), box_color, -1)
        
        # Add subtle gradient effect
        for i in range(status_box_height):
            alpha = i / status_box_height
            color = tuple(int(c * (0.7 + 0.3 * alpha)) for c in box_color)
            cv2.line(frame, (x, y - i), (x + w, y - i), color, 1)
        
        cv2.rectangle(frame, (x, y - status_box_height), 
                     (x + w, y), (255, 255, 255), 2)
        
        # Draw status text with shadow
        cv2.putText(frame, status_text, (x + 10, y - 100),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3)
        cv2.putText(frame, status_text, (x + 10, y - 100),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw alert level
        cv2.putText(frame, f"Alert Level: {alert_level}", (x + 10, y - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw driver info
        driver_text = f"Driver: {driver_name}"
        cv2.putText(frame, driver_text, (x + 10, y - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw confidence badge
        cv2.putText(frame, f"{driver_conf:.0%}", (x + w - 50, y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw biometric info box
        bio_box_height = 140
        cv2.rectangle(frame, (x, y + h), 
                     (x + w, y + h + bio_box_height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y + h), 
                     (x + w, y + h + bio_box_height), (100, 100, 100), 1)
        
        # Draw biometric data
        y_offset = y + h + 25
        
        # EAR (Eye Aspect Ratio)
        ear_color = (0, 255, 0) if ear > self.eye_closed_threshold else (0, 0, 255)
        cv2.putText(frame, f"EAR: {ear:.3f}", 
                   (x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
        y_offset += 25
        
        # MAR (Mouth Aspect Ratio)
        mar_color = (0, 255, 0) if mar < self.yawning_threshold_mar else (0, 165, 255)
        cv2.putText(frame, f"MAR: {mar:.3f}", 
                   (x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mar_color, 1)
        y_offset += 25
        
        # PERCLOS
        perclos_color = (0, 255, 0) if perclos < 0.2 else (0, 165, 255) if perclos < 0.4 else (0, 0, 255)
        cv2.putText(frame, f"PERCLOS: {perclos:.1%}", 
                   (x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, perclos_color, 1)
        y_offset += 25
        
        # Drowsiness confidence
        cv2.putText(frame, f"Drowsiness Conf: {drowsiness_conf:.1%}", 
                   (x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS and performance info in top-right corner
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        
        # Draw camera info
        camera_text = f"Camera: {'External' if self.camera_index > 0 else 'Built-in'}"
        cv2.putText(frame, camera_text, (frame.shape[1] - 120, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw blink counter
        cv2.putText(frame, f"Blinks: {self.blink_counter}", 
                   (frame.shape[1] - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw face quality indicator
        quality_color = (0, 255, 0) if face_quality > 0.7 else (0, 165, 255) if face_quality > 0.5 else (0, 0, 255)
        quality_text = f"Face Quality: {face_quality:.0%}"
        cv2.putText(frame, quality_text, (frame.shape[1] - 120, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        
        # Draw tracking status if face is lost
        if not hasattr(self, '_has_face') or not self._has_face:
            cv2.putText(frame, "SEARCHING...", (x + 10, y - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw debug info if enabled
        if self.debug_mode:
            debug_text = f"Debug: History={len(self.drowsiness_history)}"
            cv2.putText(frame, debug_text, (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def draw_alert_overlay(self, frame):
        """Draw alert overlay with enhanced visual effects"""
        if not self.alert_active:
            return
        
        alert_duration = time.time() - self.alert_start_time
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Pulsing effect
        pulse = 0.5 + 0.5 * np.sin(time.time() * 4)  # 4Hz pulse
        
        # Main alert text
        alert_text = "🚨 DROWSINESS ALERT!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_TRIPLEX, 1.8, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] // 2
        
        # Draw text with shadow and glow
        cv2.putText(frame, alert_text, (text_x + 3, text_y + 3),
                   cv2.FONT_HERSHEY_TRIPLEX, 1.8, (0, 0, 0), 5)
        cv2.putText(frame, alert_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_TRIPLEX, 1.8, (0, 0, int(255 * pulse)), 5)
        
        # Additional warning text
        warning_text = f"Driver is showing signs of drowsiness! ({alert_duration:.1f}s)"
        warning_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        warning_x = (frame.shape[1] - warning_size[0]) // 2
        
        cv2.putText(frame, warning_text, (warning_x, text_y + 60),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw warning border
        border_thickness = int(5 + 3 * pulse)
        cv2.rectangle(frame, (20, 20), (frame.shape[1]-20, frame.shape[0]-20),
                     (0, 0, int(255 * pulse)), border_thickness)
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera index {self.camera_index}")
            print("💡 Trying default camera...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Cannot open any camera")
                return
        
        # Configure camera for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n📷 Camera: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        print("\n🎥 Starting enhanced detection...")
        print("="*50)
        print("Enhanced Controls:")
        print("  • Q: Quit")
        print("  • R: Reset tracking and alerts")
        print("  • S: Toggle smoothing")
        print("  • D: Toggle debug mode (detailed info)")
        print("  • V: Toggle visual landmarks")
        print("  • C: Show registered drivers")
        print("  • +: Increase detection sensitivity")
        print("  • -: Decrease detection sensitivity")
        print("  • E: Adjust eye detection threshold")
        print("  • M: Adjust mouth detection threshold")
        print("  • P: Print detection statistics")
        print("  • A: Toggle alert sound (if enabled)")
        print("="*50)
        
        self.frame_count = 0
        self.start_time = time.time()
        smoothing = True
        box_padding = 25  # Extra padding around face box
        alert_sound = False
        
        # Performance monitoring
        processing_times = []
        
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Mirror for natural view
            
            # Calculate FPS (smoothed)
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Detect face
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb)
            
            driver_name = "Unknown"
            driver_conf = 0.0
            drowsiness_state = "Non_Drowsy"
            drowsiness_conf = 0.8
            ear_value = 0.0
            mar_value = 0.0
            perclos_value = 0.0
            face_quality = 1.0
            quality_message = "Good"
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                face_confidence = detection.score[0]
                
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                x = max(0, x - box_padding)
                y = max(0, y - box_padding)
                width = min(width + 2 * box_padding, w - x)
                height = min(height + 2 * box_padding, h - y)
                
                # Update tracking
                self.last_face_box = (x, y, width, height)
                self.face_lost_count = 0
                self._has_face = True
                
                # Extract face ROI
                if width > 60 and height > 60:
                    face_roi = frame[y:y+height, x:x+width]
                    
                    if face_roi.size > 0 and face_roi.shape[0] > 30 and face_roi.shape[1] > 30:
                        try:
                            # Assess face quality
                            face_quality, quality_message = self.assess_face_quality(face_roi, bbox)
                            
                            if face_quality >= self.face_quality_threshold:
                                # Extract features and recognize driver
                                features = self.extract_face_features(bbox, frame)
                                driver_name, driver_conf = self.recognize_driver(features)
                                
                                # Detect drowsiness with multiple methods
                                drowsiness_state, drowsiness_conf, ear_value, mar_value, perclos_value = self.detect_drowsiness(face_roi)
                                
                                # Update history for smoothing
                                self.update_history(driver_name, driver_conf, 
                                                  drowsiness_state, drowsiness_conf, 
                                                  ear_value, mar_value, perclos_value)
                                
                                # Get smoothed results
                                if smoothing and len(self.drowsiness_history) >= 3:
                                    driver_name, driver_conf, drowsiness_state, drowsiness_conf, \
                                    ear_value, mar_value, perclos_value = self.get_smoothed_results()
                                
                                # Update alert system
                                if drowsiness_state == "Drowsy":
                                    self.consecutive_drowsy_frames += 1
                                    self.consecutive_yawning_frames = 0
                                    
                                    if self.consecutive_drowsy_frames >= self.drowsy_threshold:
                                        if not self.alert_active:
                                            self.alert_active = True
                                            self.alert_start_time = time.time()
                                            self.detection_stats['alerts_triggered'] += 1
                                            print(f"🚨 DROWSINESS ALERT: {driver_name} shows drowsy signs!")
                                            print(f"   PERCLOS: {perclos_value:.1%}, EAR: {ear_value:.3f}")
                                            
                                            if alert_sound:
                                                # You can add sound alert here
                                                print("🔊 Alert sound triggered")
                                
                                elif drowsiness_state == "Yawning":
                                    self.consecutive_yawning_frames += 1
                                    self.consecutive_drowsy_frames = 0
                                    
                                    if self.consecutive_yawning_frames >= self.yawning_threshold:
                                        if not self.alert_active and time.time() - self.alert_start_time > self.alert_cooldown:
                                            self.alert_active = True
                                            self.alert_start_time = time.time()
                                            print(f"⚠️ YAWNING ALERT: {driver_name} is yawning frequently!")
                                            print(f"   MAR: {mar_value:.3f}")
                                
                                else:
                                    self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 1)
                                    self.consecutive_yawning_frames = max(0, self.consecutive_yawning_frames - 1)
                                    
                                    if self.consecutive_drowsy_frames == 0 and self.consecutive_yawning_frames == 0:
                                        self.alert_active = False
                            
                            else:
                                if self.debug_mode:
                                    print(f"⚠️ Low face quality: {quality_message} ({face_quality:.2f})")
                                
                        except Exception as e:
                            if self.debug_mode:
                                print(f"⚠️ Processing error: {e}")
            else:
                # Face lost - use last known position
                self._has_face = False
                if self.last_face_box is not None and self.face_lost_count < self.max_face_lost:
                    self.face_lost_count += 1
                    x, y, width, height = self.last_face_box
                else:
                    self.last_face_box = None
            
            # Update detection statistics
            self.detection_stats['total_frames'] += 1
            if drowsiness_state == "Drowsy":
                self.detection_stats['drowsy_frames'] += 1
            elif drowsiness_state == "Yawning":
                self.detection_stats['yawning_frames'] += 1
            
            # Draw detection info if we have a face box
            if self.last_face_box is not None:
                x, y, width, height = self.last_face_box
                self.draw_enhanced_info(frame, x, y, width, height,
                                       driver_name, driver_conf,
                                       drowsiness_state, drowsiness_conf,
                                       ear_value, mar_value, perclos_value,
                                       fps, face_quality)
            
            # Draw alert overlay if active
            self.draw_alert_overlay(frame)
            
            # Show frame
            window_name = f"Enhanced Driver Drowsiness Detection - {actual_width}x{actual_height}"
            cv2.imshow(window_name, frame)
            
            # Calculate processing time
            processing_time = time.time() - frame_start
            processing_times.append(processing_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('r'):
                self.last_face_box = None
                self.driver_history = []
                self.drowsiness_history = []
                self.face_lost_count = 0
                self.consecutive_drowsy_frames = 0
                self.consecutive_yawning_frames = 0
                self.alert_active = False
                self.blink_counter = 0
                self.blink_history = []
                self.eye_state_history = []
                print("🔄 System reset: Tracking and alerts cleared")
            elif key == ord('s'):
                smoothing = not smoothing
                print(f"🔄 Smoothing: {'ON' if smoothing else 'OFF'}")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('v'):
                self.visual_debug = not self.visual_debug
                print(f"👁️ Visual landmarks: {'ON' if self.visual_debug else 'OFF'}")
            elif key == ord('c') and self.driver_names:
                print("\n📋 Registered drivers:")
                for driver_id, name in self.driver_names.items():
                    print(f"  • {name} ({driver_id})")
            elif key == ord('+'):
                self.drowsiness_threshold = min(self.drowsiness_threshold + 0.05, 0.95)
                print(f"📈 Detection sensitivity decreased: {self.drowsiness_threshold:.2f}")
            elif key == ord('-'):
                self.drowsiness_threshold = max(self.drowsiness_threshold - 0.05, 0.3)
                print(f"📉 Detection sensitivity increased: {self.drowsiness_threshold:.2f}")
            elif key == ord('e'):
                self.eye_closed_threshold = 0.18 if self.eye_closed_threshold == 0.22 else 0.22
                print(f"👁️ Eye closed threshold: {self.eye_closed_threshold}")
            elif key == ord('m'):
                self.yawning_threshold_mar = 0.6 if self.yawning_threshold_mar == 0.65 else 0.65
                print(f"👄 Yawning threshold: {self.yawning_threshold_mar}")
            elif key == ord('p'):
                print(f"\n📊 Detection Statistics:")
                print(f"  Total frames: {self.detection_stats['total_frames']}")
                print(f"  Drowsy frames: {self.detection_stats['drowsy_frames']} ({self.detection_stats['drowsy_frames']/max(1, self.detection_stats['total_frames']):.1%})")
                print(f"  Yawning frames: {self.detection_stats['yawning_frames']} ({self.detection_stats['yawning_frames']/max(1, self.detection_stats['total_frames']):.1%})")
                print(f"  Blinks detected: {self.detection_stats['blinks']}")
                print(f"  Alerts triggered: {self.detection_stats['alerts_triggered']}")
                print(f"  Avg processing time: {avg_processing_time*1000:.1f}ms")
            elif key == ord('a'):
                alert_sound = not alert_sound
                print(f"🔊 Alert sound: {'ON' if alert_sound else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print comprehensive summary
        print(f"\n" + "="*80)
        print("✅ ENHANCED DETECTION SESSION ENDED")
        print("="*80)
        
        total_duration = time.time() - self.start_time
        avg_fps = self.frame_count / total_duration if total_duration > 0 else 0
        
        print(f"\n📊 Performance Summary:")
        print(f"  • Total frames processed: {self.frame_count}")
        print(f"  • Duration: {total_duration:.1f} seconds")
        print(f"  • Average FPS: {avg_fps:.1f}")
        print(f"  • Final FPS: {fps:.1f}")
        
        print(f"\n🚨 Alert Statistics:")
        print(f"  • Drowsy frames: {self.detection_stats['drowsy_frames']} ({self.detection_stats['drowsy_frames']/max(1, self.frame_count):.1%})")
        print(f"  • Yawning frames: {self.detection_stats['yawning_frames']} ({self.detection_stats['yawning_frames']/max(1, self.frame_count):.1%})")
        print(f"  • Total blinks: {self.detection_stats['blinks']}")
        print(f"  • Alerts triggered: {self.detection_stats['alerts_triggered']}")
        
        if self.detection_stats['drowsy_frames'] > 0:
            print(f"\n⚠️ Drowsiness detected in {self.detection_stats['drowsy_frames']/max(1, self.frame_count):.1%} of frames")
            print("   Consider reviewing driver fatigue levels")
        
        print(f"\n👋 Enhanced system shutdown complete")

def main():
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Create and run enhanced system
    system = DriverDrowsinessSystem()
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Detection stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Enhanced system shutdown complete")

if __name__ == "__main__":
    main()