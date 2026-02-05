"""
ENHANCED DROWSINESS DETECTION WITH ARDUINO COMMUNICATION
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
import serial
import threading
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("🚗 DRIVER DROWSINESS DETECTION WITH ARDUINO COMMUNICATION")
print("="*80)

class DrowsinessArduinoSystem:
    def __init__(self, arduino_port='/dev/ttyAMA0', baudrate=9600):
        # Initialize serial communication with Arduino
        self.arduino_connected = False
        self.arduino_port = arduino_port
        self.baudrate = baudrate
        self.serial_lock = threading.Lock()
        
        # Try to connect to Arduino
        self.connect_to_arduino()
        
        # Get correct paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.models_dir = os.path.join(project_root, "models")
        
        print(f"📁 Project root: {project_root}")
        print(f"📁 Models directory: {self.models_dir}")
        
        # Detection parameters
        self.debug_mode = False
        self.camera_index = self.select_camera()
        
        print("\n🔍 Loading models...")
        
        # Load driver recognition model
        try:
            mapping_path = os.path.join(self.models_dir, "driver_mapping.json")
            with open(mapping_path, 'r') as f:
                self.driver_mapping = json.load(f)
            
            self.driver_names = self.driver_mapping['driver_names']
            self.driver_model = joblib.load(os.path.join(self.models_dir, "driver_model.pkl"))
            self.driver_encoder = joblib.load(os.path.join(self.models_dir, "driver_encoder.pkl"))
            self.driver_scaler = joblib.load(os.path.join(self.models_dir, "driver_scaler.pkl"))
            
            print(f"✅ Driver model loaded")
            print(f"   Registered drivers: {list(self.driver_names.values())}")
            
        except Exception as e:
            print(f"⚠️ Driver model error: {e}")
            self.driver_model = None
            self.driver_names = {}
        
        # Load drowsiness model
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            drowsiness_model_path = os.path.join(self.models_dir, "drowsiness_model.h5")
            if not os.path.exists(drowsiness_model_path):
                drowsiness_model_path = os.path.join(self.models_dir, "drowsiness_model_enhanced.h5")
            
            if os.path.exists(drowsiness_model_path):
                self.drowsiness_model = tf.keras.models.load_model(
                    drowsiness_model_path,
                    compile=False
                )
                
                drowsiness_classes_path = os.path.join(self.models_dir, "drowsiness_classes.json")
                if not os.path.exists(drowsiness_classes_path):
                    drowsiness_classes_path = os.path.join(self.models_dir, "drowsiness_classes_enhanced.json")
                
                if os.path.exists(drowsiness_classes_path):
                    with open(drowsiness_classes_path, 'r') as f:
                        drowsiness_info = json.load(f)
                    class_indices = drowsiness_info['class_indices']
                    self.drowsiness_classes = {v: k for k, v in class_indices.items()}
                    
                    if 'input_shape' in drowsiness_info:
                        self.drowsiness_input_shape = drowsiness_info['input_shape']
                    else:
                        self.drowsiness_input_shape = self.drowsiness_model.input_shape[1:4]
                    
                    print(f"✅ Drowsiness model loaded")
                else:
                    self.drowsiness_classes = {0: "Drowsy", 1: "Non_Drowsy", 2: "Yawning"}
                    self.drowsiness_input_shape = [224, 224, 3]
            else:
                raise FileNotFoundError("No drowsiness model found")
                
        except Exception as e:
            print(f"❌ Drowsiness model error: {e}")
            self.drowsiness_model = None
            self.drowsiness_classes = {0: "Drowsy", 1: "Non_Drowsy", 2: "Yawning"}
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        
        # State tracking
        self.current_driver = "Unknown"
        self.current_status = "Non_Drowsy"
        self.current_bpm = 0  # Will be received from Arduino
        self.current_confidence = 0.0
        
        # Alert tracking
        self.alert_active = False
        self.alert_start_time = 0
        self.consecutive_drowsy_frames = 0
        self.drowsy_threshold = 10
        self.alert_cooldown = 5.0
        
        # Communication variables
        self.last_arduino_update = 0
        self.update_interval = 1.0  # Update Arduino every 1 second
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_stats = {
            'total_frames': 0,
            'drowsy_frames': 0,
            'yawning_frames': 0,
            'alerts_triggered': 0
        }
        
        print(f"\n📊 System initialized with camera index: {self.camera_index}")
        print("✅ Ready for detection with Arduino communication")
    
    def connect_to_arduino(self):
        """Establish serial connection with Arduino"""
        try:
            # Try multiple possible ports
            possible_ports = ['/dev/ttyAMA0', '/dev/ttyACM0', '/dev/ttyUSB0', 'COM3', 'COM4', 'COM5']
            
            for port in possible_ports:
                try:
                    print(f"🔌 Trying to connect to Arduino on {port}...")
                    self.ser = serial.Serial(
                        port=port,
                        baudrate=self.baudrate,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        timeout=1
                    )
                    
                    # Test connection
                    time.sleep(2)  # Wait for Arduino to initialize
                    self.ser.write(b"GET_STATUS\n")
                    response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if response and "STATUS:" in response:
                        print(f"✅ Connected to Arduino on {port}")
                        self.arduino_connected = True
                        self.arduino_port = port
                        
                        # Send initial configuration
                        self.send_to_arduino("SYSTEM_START:Drowsiness Detection Active")
                        return True
                    else:
                        self.ser.close()
                        
                except Exception as e:
                    if 'COM' in port:
                        continue
                    print(f"⚠️ Failed on {port}: {e}")
            
            print("❌ Could not connect to Arduino. Running in standalone mode.")
            self.arduino_connected = False
            return False
            
        except Exception as e:
            print(f"❌ Arduino connection error: {e}")
            self.arduino_connected = False
            return False
    
    def send_to_arduino(self, message):
        """Send message to Arduino"""
        if not self.arduino_connected:
            return False
        
        try:
            with self.serial_lock:
                self.ser.write(f"{message}\n".encode())
                if self.debug_mode:
                    print(f"📤 To Arduino: {message}")
            return True
        except Exception as e:
            print(f"❌ Failed to send to Arduino: {e}")
            self.arduino_connected = False
            return False
    
    def read_from_arduino(self):
        """Read data from Arduino"""
        if not self.arduino_connected:
            return None
        
        try:
            with self.serial_lock:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if data:
                        if self.debug_mode:
                            print(f"📥 From Arduino: {data}")
                        return data
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Failed to read from Arduino: {e}")
        
        return None
    
    def select_camera(self):
        """Select camera"""
        for camera_index in [1, 2, 3, 0]:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return camera_index
        return 0
    
    def extract_face_features(self, bbox, frame):
        """Extract face features"""
        h, w = frame.shape[:2]
        features = [
            bbox.xmin, bbox.ymin, bbox.width, bbox.height,
            bbox.xmin * w, bbox.ymin * h, bbox.width * w, bbox.height * h,
            bbox.xmin + bbox.width/2, bbox.ymin + bbox.height/2,
            w / max(h, 1), bbox.width / max(bbox.height, 0.001)
        ]
        return np.array(features, dtype=np.float32)
    
    def recognize_driver(self, features):
        """Recognize driver"""
        if self.driver_model is None:
            return "Unknown", 0.5
        
        try:
            features = features.reshape(1, -1)
            
            if hasattr(self.driver_scaler, 'mean_'):
                n_expected = self.driver_scaler.mean_.shape[0]
                if features.shape[1] > n_expected:
                    features = features[:, :n_expected]
                elif features.shape[1] < n_expected:
                    padding = np.zeros((1, n_expected - features.shape[1]))
                    features = np.hstack([features, padding])
            
            features_scaled = self.driver_scaler.transform(features)
            
            if hasattr(self.driver_model, 'predict_proba'):
                proba = self.driver_model.predict_proba(features_scaled)[0]
                pred_idx = np.argmax(proba)
                confidence = proba[pred_idx]
            else:
                pred_idx = self.driver_model.predict(features_scaled)[0]
                confidence = 0.8
            
            driver_id = self.driver_encoder.inverse_transform([pred_idx])[0]
            driver_id_str = str(driver_id)
            driver_name = self.driver_names.get(driver_id_str, f"Driver_{pred_idx}")
            
            return driver_name, confidence
            
        except Exception as e:
            return "Unknown", 0.0
    
    def detect_drowsiness(self, face_roi):
        """Detect drowsiness state"""
        if self.drowsiness_model is None:
            return "Non_Drowsy", 0.7
        
        try:
            target_size = (self.drowsiness_input_shape[0], self.drowsiness_input_shape[1])
            img = cv2.resize(face_roi, target_size)
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            predictions = self.drowsiness_model.predict(img, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            if class_idx in self.drowsiness_classes:
                state = self.drowsiness_classes[class_idx]
            else:
                states = ["Drowsy", "Non_Drowsy", "Yawning"]
                state = states[min(class_idx, len(states)-1)]
            
            if confidence < 0.65:
                state = "Non_Drowsy"
            
            return state, confidence
            
        except Exception as e:
            return "Non_Drowsy", 0.7
    
    def process_arduino_data(self, data):
        """Process data received from Arduino"""
        if not data:
            return
        
        # Parse BPM data
        if "HR:" in data or "BPM:" in data:
            try:
                if "HR:" in data:
                    bpm_str = data.split("HR:")[1].strip()
                else:
                    bpm_str = data.split("BPM:")[1].strip()
                
                if bpm_str.isdigit():
                    self.current_bpm = int(bpm_str)
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Failed to parse BPM: {e}")
        
        # Parse status data
        elif "STATUS:" in data:
            try:
                status = data.split("STATUS:")[1].strip()
                # Could update display based on Arduino status
            except:
                pass
    
    def update_arduino(self):
        """Update Arduino with current status"""
        current_time = time.time()
        
        if current_time - self.last_arduino_update < self.update_interval:
            return
        
        # Prepare status message for Arduino
        if self.current_status == "Drowsy":
            status_code = "DROWSY"
            alert_level = "HIGH"
        elif self.current_status == "Yawning":
            status_code = "YAWNING"
            alert_level = "MEDIUM"
        else:
            status_code = "NORMAL"
            alert_level = "LOW"
        
        # Create message
        message = f"STATUS:{status_code}:{self.current_driver}:{alert_level}"
        if self.alert_active:
            message += ":ALERT"
        
        # Send to Arduino
        self.send_to_arduino(message)
        self.last_arduino_update = current_time
        
        # Request BPM from Arduino periodically
        if current_time % 10 < 0.1:  # Every ~10 seconds
            self.send_to_arduino("GET_BPM")
    
    def handle_alert(self):
        """Handle drowsiness alert"""
        if self.current_status == "Drowsy":
            self.consecutive_drowsy_frames += 1
            
            if self.consecutive_drowsy_frames >= self.drowsy_threshold:
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = time.time()
                    self.detection_stats['alerts_triggered'] += 1
                    
                    print(f"🚨 DROWSINESS ALERT: {self.current_driver}")
                    
                    # Send urgent alert to Arduino
                    self.send_to_arduino(f"ALERT:DROWSY:{self.current_driver}")
                    
                    # Here you could trigger additional actions:
                    # - Sound alarm
                    # - Send notification
                    # - Log incident
        else:
            self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 1)
            
            if self.consecutive_drowsy_frames == 0 and self.alert_active:
                if time.time() - self.alert_start_time > self.alert_cooldown:
                    self.alert_active = False
                    self.send_to_arduino("ALERT:CLEAR")
    
    def draw_interface(self, frame, driver_name, status, confidence, fps):
        """Draw detection interface"""
        h, w = frame.shape[:2]
        
        # Status colors
        if status == "Drowsy":
            color = (0, 0, 255)  # Red
            status_text = "🚨 DROWSY"
        elif status == "Yawning":
            color = (0, 165, 255)  # Orange
            status_text = "⚠️ YAWNING"
        else:
            color = (0, 255, 0)  # Green
            status_text = "✅ ALERT"
        
        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
        cv2.rectangle(frame, (0, 0), (w, 60), (255, 255, 255), 2)
        
        # Status text
        cv2.putText(frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        # Driver info
        driver_text = f"Driver: {driver_name}"
        cv2.putText(frame, driver_text, (w - 300, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Arduino connection status
        arduino_status = "✅ ARDUINO" if self.arduino_connected else "❌ ARDUINO"
        arduino_color = (0, 255, 0) if self.arduino_connected else (0, 0, 255)
        cv2.putText(frame, arduino_status, (20, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, arduino_color, 2)
        
        # Heart rate from Arduino
        bpm_text = f"HR: {self.current_bpm} BPM" if self.current_bpm > 0 else "HR: --- BPM"
        cv2.putText(frame, bpm_text, (w - 200, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 100, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Confidence
        conf_text = f"Conf: {confidence:.0%}"
        cv2.putText(frame, conf_text, (w - 150, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def draw_alert_overlay(self, frame):
        """Draw alert overlay"""
        if not self.alert_active:
            return
        
        alert_duration = time.time() - self.alert_start_time
        
        # Create red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Pulsing effect
        pulse = 0.5 + 0.5 * np.sin(time.time() * 4)
        
        # Alert text
        alert_text = "🚨 DROWSINESS DETECTED!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 5)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] // 2
        
        cv2.putText(frame, alert_text, (text_x + 3, text_y + 3),
                   cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(frame, alert_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, int(255 * pulse)), 5)
        
        # Driver info
        driver_text = f"Driver: {self.current_driver} - Duration: {alert_duration:.1f}s"
        driver_size = cv2.getTextSize(driver_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
        driver_x = (frame.shape[1] - driver_size[0]) // 2
        
        cv2.putText(frame, driver_text, (driver_x, text_y + 50),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    def run(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n🎥 Starting detection with Arduino communication...")
        print("="*50)
        print("Controls:")
        print("  • Q: Quit")
        print("  • R: Reset system")
        print("  • D: Toggle debug mode")
        print("  • S: Test Arduino connection")
        print("  • A: Manually trigger alert")
        print("="*50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Read from Arduino (non-blocking)
            arduino_data = self.read_from_arduino()
            if arduino_data:
                self.process_arduino_data(arduino_data)
            
            # Detect face
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb)
            
            driver_name = "Unknown"
            status = "Non_Drowsy"
            confidence = 0.0
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                width = min(width + 2 * padding, w - x)
                height = min(height + 2 * padding, h - y)
                
                # Extract face ROI
                if width > 60 and height > 60:
                    face_roi = frame[y:y+height, x:x+width]
                    
                    if face_roi.size > 0:
                        # Extract features
                        features = self.extract_face_features(bbox, frame)
                        driver_name, driver_conf = self.recognize_driver(features)
                        
                        # Detect drowsiness
                        status, drowsiness_conf = self.detect_drowsiness(face_roi)
                        confidence = (driver_conf + drowsiness_conf) / 2
                        
                        # Draw face box
                        color = (0, 0, 255) if status == "Drowsy" else (0, 165, 255) if status == "Yawning" else (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
                        
                        # Update current state
                        self.current_driver = driver_name
                        self.current_status = status
                        self.current_confidence = confidence
            
            # Update detection stats
            self.detection_stats['total_frames'] += 1
            if status == "Drowsy":
                self.detection_stats['drowsy_frames'] += 1
            elif status == "Yawning":
                self.detection_stats['yawning_frames'] += 1
            
            # Handle alerts
            self.handle_alert()
            
            # Update Arduino
            self.update_arduino()
            
            # Draw interface
            self.draw_interface(frame, driver_name, status, confidence, fps)
            
            # Draw alert overlay if active
            self.draw_alert_overlay(frame)
            
            # Show frame
            cv2.imshow("Drowsiness Detection with Arduino", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.current_driver = "Unknown"
                self.current_status = "Non_Drowsy"
                self.alert_active = False
                self.consecutive_drowsy_frames = 0
                print("🔄 System reset")
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"🐛 Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('s'):
                # Test Arduino connection
                if self.arduino_connected:
                    self.send_to_arduino("GET_STATUS")
                    print("📤 Sent test command to Arduino")
                else:
                    print("❌ Arduino not connected")
            elif key == ord('a'):
                # Manually trigger alert
                self.alert_active = True
                self.alert_start_time = time.time()
                self.send_to_arduino("ALERT:TEST:Manual Alert")
                print("🚨 Manual alert triggered")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if self.arduino_connected:
            self.send_to_arduino("SYSTEM_STOP:Detection Ended")
            self.ser.close()
        
        # Print summary
        print(f"\n" + "="*80)
        print("📊 DETECTION SESSION SUMMARY")
        print("="*80)
        
        total_duration = time.time() - self.start_time
        avg_fps = self.frame_count / total_duration if total_duration > 0 else 0
        
        print(f"\n📈 Performance:")
        print(f"  • Total frames: {self.frame_count}")
        print(f"  • Duration: {total_duration:.1f}s")
        print(f"  • Average FPS: {avg_fps:.1f}")
        
        print(f"\n🚨 Detection Results:")
        print(f"  • Drowsy frames: {self.detection_stats['drowsy_frames']} ({self.detection_stats['drowsy_frames']/max(1, self.frame_count):.1%})")
        print(f"  • Yawning frames: {self.detection_stats['yawning_frames']} ({self.detection_stats['yawning_frames']/max(1, self.frame_count):.1%})")
        print(f"  • Alerts triggered: {self.detection_stats['alerts_triggered']}")
        
        print(f"\n🔌 Arduino Communication:")
        print(f"  • Connected: {'Yes' if self.arduino_connected else 'No'}")
        print(f"  • Port: {self.arduino_port if self.arduino_connected else 'N/A'}")
        
        print(f"\n👋 System shutdown complete")

def main():
    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # For Windows, use COM port
    if os.name == 'nt':
        arduino_port = 'COM3'  # Change to your COM port
    else:
        arduino_port = '/dev/ttyAMA0'  # For Raspberry Pi GPIO
    
    # Create and run system
    system = DrowsinessArduinoSystem(arduino_port=arduino_port, baudrate=9600)
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Detection stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()