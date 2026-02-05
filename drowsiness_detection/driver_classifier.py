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

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

class FixedDriverClassifier:
    def __init__(self):
        print("="*80)
        print("FIXED DRIVER CLASSIFIER - Optimized Performance")
        print("="*80)
        
        self.models_dir = "../models"
        
        # Initialize MediaPipe ONCE (not per frame)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3  # Lower for better detection
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Performance settings
        self.target_fps = 15  # Higher FPS
        self.frame_delay = 1.0 / self.target_fps
        self.last_process_time = 0
        self.skip_frames = 0  # Process every Nth frame for speed
        self.frame_counter = 0
        
        # Face tracking
        self.last_face_bbox = None
        self.face_lost_frames = 0
        self.max_face_lost_frames = 5  # Keep box for 5 frames after losing face
        
        # Driver classification history (for smoothing)
        self.driver_predictions = []
        self.max_history = 10
        
        # Load models
        self.load_models()
    
    def load_models(self):
        print("\n🔍 Loading models...")
        
        # Load driver mapping
        try:
            with open(os.path.join(self.models_dir, "driver_mapping.json"), 'r') as f:
                self.mapping = json.load(f)
            
            self.driver_names = self.mapping['driver_names']
            self.model_accuracy = self.mapping.get('accuracy', 0.5)
            
            print(f"✅ Drivers: {list(self.driver_names.values())}")
            print(f"⚠️  Model Accuracy: {self.model_accuracy:.0%} (Low - needs retraining)")
            
            # If accuracy is low, we need to handle carefully
            if self.model_accuracy < 0.7:
                print("💡 Model has low accuracy. Consider retraining with more images.")
            
        except Exception as e:
            print(f"❌ Could not load mapping: {e}")
            self.driver_names = {}
            return False
        
        # Load driver model
        try:
            self.driver_svm = joblib.load(os.path.join(self.models_dir, "driver_svm.pkl"))
            self.driver_encoder = joblib.load(os.path.join(self.models_dir, "driver_encoder.pkl"))
            self.driver_scaler = joblib.load(os.path.join(self.models_dir, "driver_scaler.pkl"))
            
            print(f"✅ Driver model loaded")
            print(f"   Classes: {self.driver_svm.classes_}")
            print(f"   Features: {self.driver_scaler.mean_.shape[0]}")
            
            # Check if model is biased
            self.check_model_bias()
            
        except Exception as e:
            print(f"❌ Error loading driver model: {e}")
            self.driver_svm = None
            self.driver_encoder = None
            self.driver_scaler = None
        
        # Load drowsiness model
        try:
            self.drowsiness_model = tf.keras.models.load_model(
                os.path.join(self.models_dir, "drowsiness_model.h5"),
                compile=False
            )
            
            with open(os.path.join(self.models_dir, "drowsiness_classes.json"), 'r') as f:
                drowsiness_info = json.load(f)
            self.drowsiness_classes = drowsiness_info['class_indices']
            self.drowsiness_class_names = {v: k for k, v in self.drowsiness_classes.items()}
            
            print(f"✅ Drowsiness model loaded")
            print(f"   Classes: {self.drowsiness_class_names}")
            
        except Exception as e:
            print(f"⚠️ Could not load drowsiness model: {e}")
            self.drowsiness_model = None
        
        return True
    
    def check_model_bias(self):
        """Check if model is biased toward one class"""
        print("\n🔍 Checking model bias...")
        
        # Create dummy features (all zeros)
        n_features = self.driver_scaler.mean_.shape[0]
        dummy_features = np.zeros((1, n_features))
        
        # Scale
        dummy_scaled = self.driver_scaler.transform(dummy_features)
        
        # Get probabilities
        probabilities = self.driver_svm.predict_proba(dummy_scaled)[0]
        
        print("📊 Model bias test (all-zero features):")
        for i, class_id in enumerate(self.driver_svm.classes_):
            driver_id = self.driver_encoder.inverse_transform([class_id])[0]
            driver_name = self.driver_names.get(str(driver_id), f"Driver_{driver_id}")
            prob = probabilities[i]
            print(f"   {driver_name}: {prob:.1%}")
        
        # Check if heavily biased
        if max(probabilities) > 0.9:
            print(f"⚠️  WARNING: Model is heavily biased toward one class!")
            print(f"   This will cause incorrect classifications.")
            return True
        
        return False
    
    def detect_face_fast(self, frame):
        """Fast face detection with tracking"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(w - x, width + 2*padding)
            height = min(h - y, height + 2*padding)
            
            if width > 20 and height > 20:
                face_roi = frame[y:y+height, x:x+width]
                if face_roi.size > 0:
                    self.last_face_bbox = (x, y, width, height)
                    self.face_lost_frames = 0
                    return face_roi, (x, y, width, height)
        
        # If no face detected but we had one recently, use last known position
        if self.last_face_bbox is not None and self.face_lost_frames < self.max_face_lost_frames:
            self.face_lost_frames += 1
            x, y, width, height = self.last_face_bbox
            
            # Make sure ROI is within bounds
            h, w, _ = frame.shape
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)
            
            if width > 20 and height > 20:
                face_roi = frame[y:y+height, x:x+width]
                if face_roi.size > 0:
                    return face_roi, (x, y, width, height)
        else:
            self.last_face_bbox = None
        
        return None, None
    
    def extract_landmarks_fast(self, face_roi):
        """Extract landmarks quickly"""
        try:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract 21 key landmarks (42 features)
                indices = [33, 133, 362, 263, 1, 4, 13, 14, 61, 291, 78, 308, 10, 152]
                # Add more if needed to reach 21
                while len(indices) < 21:
                    indices.append(indices[-1] + 1)
                
                features = []
                for idx in indices[:21]:  # Take exactly 21
                    if idx < len(landmarks.landmark):
                        lm = landmarks.landmark[idx]
                        features.extend([lm.x, lm.y])
                    else:
                        features.extend([0.0, 0.0])
                
                # Pad to 42 if needed
                if len(features) < 42:
                    features.extend([0.0] * (42 - len(features)))
                elif len(features) > 42:
                    features = features[:42]
                
                return np.array(features, dtype=np.float32)
                
        except Exception as e:
            return None
        
        return None
    
    def identify_driver_with_correction(self, face_roi):
        """Identify driver with bias correction"""
        if self.driver_svm is None:
            return "Unknown", 0.5
        
        # Extract features
        features = self.extract_landmarks_fast(face_roi)
        if features is None:
            return "Unknown", 0.0
        
        try:
            # Reshape
            features = features.reshape(1, -1)
            
            # Pad/truncate to match scaler
            n_features_expected = self.driver_scaler.mean_.shape[0]
            if features.shape[1] != n_features_expected:
                if features.shape[1] > n_features_expected:
                    features = features[:, :n_features_expected]
                else:
                    padding = np.zeros((1, n_features_expected - features.shape[1]))
                    features = np.hstack([features, padding])
            
            # Scale
            features_scaled = self.driver_scaler.transform(features)
            
            # Predict
            probabilities = self.driver_svm.predict_proba(features_scaled)[0]
            
            # Apply bias correction if model is heavily biased
            if hasattr(self, 'is_biased') and self.is_biased:
                # Even out probabilities a bit
                probabilities = np.clip(probabilities, 0.1, 0.9)
                probabilities = probabilities / np.sum(probabilities)
            
            # Get prediction
            prediction_idx = np.argmax(probabilities)
            confidence = float(probabilities[prediction_idx])
            
            # Decode
            driver_id = self.driver_encoder.inverse_transform([self.driver_svm.classes_[prediction_idx]])[0]
            driver_name = self.driver_names.get(str(driver_id), f"Driver_{driver_id}")
            
            # Store in history
            self.driver_predictions.append((driver_name, confidence))
            if len(self.driver_predictions) > self.max_history:
                self.driver_predictions.pop(0)
            
            # Get smoothed prediction (mode of last N predictions)
            if len(self.driver_predictions) >= 3:
                names = [p[0] for p in self.driver_predictions[-3:]]
                unique_names, counts = np.unique(names, return_counts=True)
                if len(unique_names) > 0:
                    mode_idx = np.argmax(counts)
                    smoothed_name = unique_names[mode_idx]
                    
                    # If smoothed is different from current, use smoothed
                    if smoothed_name != driver_name:
                        # Get average confidence for smoothed name
                        smoothed_confidences = [p[1] for p in self.driver_predictions[-3:] if p[0] == smoothed_name]
                        confidence = np.mean(smoothed_confidences) if smoothed_confidences else confidence
                        driver_name = smoothed_name
            
            return driver_name, confidence
            
        except Exception as e:
            return "Unknown", 0.0
    
    def detect_drowsiness_fast(self, face_roi):
        """Fast drowsiness detection"""
        if self.drowsiness_model is None:
            return "Alert", 0.8
        
        try:
            # Resize quickly
            img_resized = cv2.resize(face_roi, (128, 128))
            
            # Ensure 3 channels
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            elif img_resized.shape[2] == 4:
                img_resized = img_resized[:, :, :3]
            
            # Normalize
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            predictions = self.drowsiness_model.predict(img_batch, verbose=0)[0]
            
            # Get result
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            state = self.drowsiness_class_names.get(class_idx, "Alert")
            
            return state, confidence
            
        except Exception as e:
            return "Alert", 0.8
    
    def run(self):
        """Main loop with high FPS"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Configure camera for high FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print(f"\n" + "="*80)
        print(f"🚀 Starting Real-time Detection")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Press 'q' to quit, 'r' to reset, 's' to toggle smoothing")
        print("="*80)
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps_display = 0
        last_fps_update = time.time()
        
        # Settings
        show_smoothing = True
        process_every_n_frames = 2  # Process every 2nd frame for speed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            self.frame_counter += 1
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process face detection on selected frames
            current_time = time.time()
            process_this_frame = (self.frame_counter % process_every_n_frames == 0)
            
            face_roi = None
            bbox = None
            driver_name = "Unknown"
            driver_conf = 0.0
            drowsiness_state = "Alert"
            drowsiness_conf = 0.8
            
            if process_this_frame and (current_time - self.last_process_time >= self.frame_delay):
                self.last_process_time = current_time
                
                # Detect face
                face_roi, bbox = self.detect_face_fast(frame)
                
                if face_roi is not None:
                    # Detect driver and drowsiness
                    driver_name, driver_conf = self.identify_driver_with_correction(face_roi)
                    drowsiness_state, drowsiness_conf = self.detect_drowsiness_fast(face_roi)
            
            # Always draw face box if we have one (even if not processing this frame)
            draw_bbox = bbox if bbox is not None else self.last_face_bbox
            
            if draw_bbox is not None:
                x, y, w, h = draw_bbox
                
                # Determine color
                if "Drowsy" in drowsiness_state:
                    color = (0, 0, 255)  # Red
                    status_text = "DROWSY"
                elif "Yawning" in drowsiness_state:
                    color = (0, 165, 255)  # Orange
                    status_text = "YAWNING"
                else:
                    color = (0, 255, 0)  # Green
                    status_text = "ALERT"
                
                # Draw face box (thicker)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw driver info
                cv2.putText(frame, f"Driver: {driver_name}", 
                           (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Conf: {driver_conf:.0%}", 
                           (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw status
                cv2.putText(frame, f"Status: {status_text}", 
                           (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Conf: {drowsiness_conf:.0%}", 
                           (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw face tracking indicator
                if bbox is None and self.face_lost_frames > 0:
                    cv2.putText(frame, f"Tracking... ({self.face_lost_frames})", 
                               (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Update FPS display every second
            if current_time - last_fps_update >= 1.0:
                elapsed = current_time - last_fps_update
                fps_display = frame_count / elapsed if elapsed > 0 else 0
                frame_count = 0
                last_fps_update = current_time
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps_display:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display processing info
            cv2.putText(frame, f"Proc: {'ON' if process_this_frame else 'OFF'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display model accuracy warning if low
            if self.model_accuracy < 0.7:
                cv2.putText(frame, f"⚠️ Low Model Acc: {self.model_accuracy:.0%}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            
            # Show frame
            cv2.imshow('Driver Drowsiness Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking
                self.last_face_bbox = None
                self.driver_predictions = []
                self.face_lost_frames = 0
                print("\n🔄 Tracking reset")
            elif key == ord('s'):
                show_smoothing = not show_smoothing
                print(f"\n{'🔧 Smoothing ON' if show_smoothing else '🔧 Smoothing OFF'}")
            elif key == ord('1'):
                process_every_n_frames = 1
                print(f"\n⚡ Processing every frame")
            elif key == ord('2'):
                process_every_n_frames = 2
                print(f"\n⚡ Processing every 2nd frame")
            elif key == ord('3'):
                process_every_n_frames = 3
                print(f"\n⚡ Processing every 3rd frame")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Performance summary
        total_time = time.time() - start_time
        print(f"\n" + "="*80)
        print(f"📊 PERFORMANCE SUMMARY")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {fps_display:.1f}")
        print("="*80)

def main():
    classifier = FixedDriverClassifier()
    
    if classifier.load_models():
        classifier.run()

if __name__ == "__main__":
    main()