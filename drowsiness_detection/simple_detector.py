
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import json
import joblib
import mediapipe as mp
import time

class SimpleDetector:
    def __init__(self):
        print("SIMPLE DROWSINESS & DRIVER DETECTOR")
        
        self.models_dir = "models"
        
        # Load driver model
        self.driver_model = joblib.load(f"{self.models_dir}/driver_svm.pkl")
        self.driver_encoder = joblib.load(f"{self.models_dir}/driver_encoder.pkl")
        self.driver_scaler = joblib.load(f"{self.models_dir}/driver_scaler.pkl")
        
        with open(f"{self.models_dir}/driver_mapping.json", "r") as f:
            self.driver_mapping = json.load(f)
        
        print(f"Drivers: {list(self.driver_mapping['driver_names'].values())}")
        
        # Load drowsiness model
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            self.drowsiness_model = tf.keras.models.load_model(
                f"{self.models_dir}/drowsiness_model.h5", compile=False
            )
            with open(f"{self.models_dir}/drowsiness_classes.json", "r") as f:
                drowsiness_info = json.load(f)
            self.drowsiness_classes = {v: k for k, v in drowsiness_info["class_indices"].items()}
            print("Drowsiness model loaded")
        except:
            print("Using dummy drowsiness detection")
            self.drowsiness_model = None
        
        # MediaPipe
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.3
        )
        
        self.last_driver = "Unknown"
        self.last_drowsiness = "Alert"
    
    def detect_face_features(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Same features as training
            features = [
                bbox.xmin, bbox.ymin, bbox.width, bbox.height,
                detection.score[0],
                bbox.xmin + bbox.width/2,
                bbox.ymin + bbox.height/2,
                bbox.width / max(bbox.height, 0.001),
                bbox.xmin * 100, bbox.ymin * 100
            ]
            
            return np.array(features), (x, y, width, height)
        
        return None, None
    
    def recognize_driver(self, features):
        features = features.reshape(1, -1)
        
        # Ensure correct feature count
        n_expected = self.driver_scaler.mean_.shape[0]
        if features.shape[1] > n_expected:
            features = features[:, :n_expected]
        elif features.shape[1] < n_expected:
            padding = np.zeros((1, n_expected - features.shape[1]))
            features = np.hstack([features, padding])
        
        features_scaled = self.driver_scaler.transform(features)
        
        if hasattr(self.driver_model, "predict_proba"):
            proba = self.driver_model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
        else:
            pred_idx = self.driver_model.predict(features_scaled)[0]
            confidence = 0.8
        
        driver_id = self.driver_encoder.inverse_transform([pred_idx])[0]
        driver_name = self.driver_mapping["driver_names"].get(str(driver_id), f"Driver_{pred_idx}")
        
        return driver_name, confidence
    
    def detect_drowsiness(self, face_roi):
        if self.drowsiness_model is None:
            return "Alert", 0.8
        
        try:
            img = cv2.resize(face_roi, (128, 128))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)
            
            predictions = self.drowsiness_model.predict(img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            state = self.drowsiness_classes.get(class_idx, "Alert")
            
            return state, confidence
        except:
            return "Alert", 0.8
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect face
            features, bbox = self.detect_face_features(frame)
            
            if features is not None:
                x, y, w, h = bbox
                face_roi = frame[y:y+h, x:x+w]
                
                # Recognize driver
                driver_name, driver_conf = self.recognize_driver(features)
                
                # Detect drowsiness
                drowsiness_state, drowsiness_conf = self.detect_drowsiness(face_roi)
                
                # Update last values
                self.last_driver = driver_name
                self.last_drowsiness = drowsiness_state
                
                # Draw
                color = (0, 0, 255) if "Drowsy" in drowsiness_state else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{driver_name} ({driver_conf:.0%})", 
                           (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"{drowsiness_state} ({drowsiness_conf:.0%})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Always show last detection
            cv2.putText(frame, f"Last: {self.last_driver} - {self.last_drowsiness}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Detector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SimpleDetector()
    detector.run()
