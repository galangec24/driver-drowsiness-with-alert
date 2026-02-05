import time
import numpy as np
from collections import deque
from datetime import datetime
import cv2
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_model import DrowsinessMLModel
from face_processor import FaceProcessor
from utils import detect_blink, calculate_blink_rate
from dashboard import create_dashboard1, create_dashboard2, driver_classifier

class AdvancedDrowsinessDetector:
    def __init__(self, driver_db_path='../backend/drivers.db'):
        """Initialize the drowsiness detection system"""
        print("\n" + "="*80)
        print("🚗 ADVANCED DRIVER DROWSINESS MONITORING SYSTEM")
        print("="*80)
        print("🔧 Initializing Face Mesh...")
        
        # Initialize components
        self.face_processor = FaceProcessor()
        
        print("🔧 Initializing ML Model...")
        self.ml_model = DrowsinessMLModel(models_dir='../models')
        
        # Load thresholds
        self.load_thresholds()
        
        # Frame counters for sustained detection
        self.CONSEC_FRAMES_EYE = 15
        self.CONSEC_FRAMES_MOUTH = 10
        self.BLINK_FRAMES = 5
        
        # State variables
        self.eye_counter = 0
        self.mouth_counter = 0
        self.blink_counter = 0
        self.status = "NORMAL"
        self.alert_active = False
        self.prev_ear = 0.25
        self.face_detected = False
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.total_blinks = 0
        self.last_blink_time = 0
        self.blink_start_frame = 0
        self.blink_end_frame = 0
        
        self.ear_history = deque(maxlen=100)
        self.mar_history = deque(maxlen=100)
        self.status_history = deque(maxlen=100)
        self.blink_history = deque(maxlen=100)
        
        # ML prediction results
        self.ml_accuracy = 0.0
        self.ml_confidence = 0.0
        
        # Normal behavior baseline
        self.blink_rate_normal = 15
        self.current_blink_rate = 0
        
        # Driver information
        self.driver_name = "Unknown"
        self.driver_id = None
        self.driver_confidence = 0.0
        
        # Camera selection
        self.camera_index = self.detect_and_select_camera()
        
        # Dashboard data
        self.dashboard_data = {
            'ear': 0.0,
            'mar': 0.0,
            'status': "N/A",
            'blinks': 0,
            'fps': 0,
            'timestamp': "",
            'face_detected': False,
            'ml_enabled': False,
            'ml_accuracy': 0.0,
            'ml_confidence': 0.0,
            'driver_name': "Unknown",
            'driver_confidence': 0.0
        }
        
        # Create display windows
        self.setup_windows()
        
        # Display system configuration
        self.display_configuration()
        
        # Print driver classifier status
        self.print_driver_classifier_status()
    
    def load_thresholds(self):
        """Load thresholds from ML model or use defaults"""
        if self.ml_model.thresholds:
            self.EYE_AR_THRESH = self.ml_model.thresholds.get('EYE_AR_THRESH', 0.22)
            self.MOUTH_AR_THRESH = self.ml_model.thresholds.get('MOUTH_AR_THRESH', 0.55)
            self.BLINK_THRESH = self.ml_model.thresholds.get('BLINK_THRESH', 0.18)
            print("✅ Using thresholds from ML model")
        else:
            # Default thresholds
            self.EYE_AR_THRESH = 0.22
            self.MOUTH_AR_THRESH = 0.55
            self.BLINK_THRESH = 0.18
            print("⚠️ Using default thresholds")
    
    def detect_and_select_camera(self):
        """Detect available cameras and select the best one"""
        print("\n🔍 Scanning for available cameras...")
        available_cameras = []
        
        for camera_idx in range(5):
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    available_cameras.append(camera_idx)
                    cam_type = "Built-in" if camera_idx == 0 else f"External"
                    print(f"  ✅ {cam_type} camera found at index {camera_idx}")
        
        if not available_cameras:
            print("❌ No cameras detected!")
            print("💡 Please connect a camera and restart the program")
            return 0
        
        # Strategy: Prefer external cameras (index > 0)
        for cam_idx in available_cameras:
            if cam_idx > 0:
                print(f"✅ Selected external camera (index {cam_idx})")
                return cam_idx
        
        # Fallback to built-in camera
        print(f"✅ Selected built-in camera (index 0)")
        return 0
    
    def setup_windows(self):
        """Setup display windows for dashboards"""
        try:
            cv2.namedWindow('DASHBOARD 1: Live Monitoring', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('DASHBOARD 1: Live Monitoring', 800, 600)
            cv2.moveWindow('DASHBOARD 1: Live Monitoring', 50, 50)
            
            cv2.namedWindow('DASHBOARD 2: Analytics Dashboard', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('DASHBOARD 2: Analytics Dashboard', 800, 600)
            cv2.moveWindow('DASHBOARD 2: Analytics Dashboard', 900, 50)
            
            print("✅ Windows created successfully")
        except cv2.error as e:
            print(f"⚠️ Window creation error: {e}")
    
    def display_configuration(self):
        """Display system configuration"""
        print("\n" + "="*80)
        print("⚙️ SYSTEM CONFIGURATION")
        print("="*80)
        print(f"📊 Detection Thresholds:")
        print(f"Eye (EAR): {self.EYE_AR_THRESH}")
        print(f"Mouth (MAR): {self.MOUTH_AR_THRESH}")
        print(f"Blink: {self.BLINK_THRESH}")
        print(f"\n🤖 Drowsiness ML Model: {'✅ Enabled' if self.ml_model.is_available else '❌ Disabled'}")
        if self.ml_model.is_available:
            print(f"Training Accuracy: {self.ml_model.accuracy:.1%}")
            print(f"Features: {self.ml_model.num_features}")
        
        # Camera info
        print(f"\n📷 Camera: {'External' if self.camera_index > 0 else 'Built-in'} (Index: {self.camera_index})")
        print("="*80)
    
    def print_driver_classifier_status(self):
        """Print driver classifier status"""
        print("\n👤 DRIVER RECOGNITION STATUS")
        print("-" * 40)
        
        if hasattr(driver_classifier, 'model_loaded'):
            if driver_classifier.model_loaded:
                print("✅ Driver Recognition: ACTIVE")
                if hasattr(driver_classifier, 'driver_names'):
                    driver_count = len(driver_classifier.driver_names)
                    print(f"   • Registered Drivers: {driver_count}")
                    if driver_count > 0:
                        print(f"   • Driver Names: {list(driver_classifier.driver_names.values())}")
                if hasattr(driver_classifier, 'current_driver_name'):
                    print(f"   • Current Driver: {driver_classifier.current_driver_name}")
                print(f"   • Recognition Interval: {driver_classifier.recognition_interval}s")
            else:
                print("⚠️ Driver Recognition: INACTIVE")
                print("   • Status: Model not loaded or training required")
                print("   • Action: Run 'python train_driver_classifier.py'")
        else:
            print("❌ Driver Recognition: NOT AVAILABLE")
            print("   • driver_classifier module not found")
        
        print("-" * 40)
    
    def calculate_ear_from_points(self, eye_points):
        """Calculate EAR from eye landmark points"""
        if len(eye_points) < 6:
            return 0.25
        
        p1, p2, p3, p4, p5, p6 = eye_points[:6]
        p1, p2, p3, p4, p5, p6 = map(np.array, [p1, p2, p3, p4, p5, p6])
        
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        
        if C == 0:
            return 0.25
        
        return (A + B) / (2.0 * C)
    
    def detect_state(self, ear, mar, prev_ear):
        """Advanced state detection with improved blink detection"""
        new_status = "NORMAL"
        blink_detected = detect_blink(ear, prev_ear, self.BLINK_THRESH, self)
        
        # Calculate blink rate
        self.current_blink_rate = calculate_blink_rate(self.blink_history)
        
        # EYE DETECTION (Drowsiness)
        if ear < self.EYE_AR_THRESH:
            self.eye_counter += 1
            if self.eye_counter >= self.CONSEC_FRAMES_EYE:
                new_status = "DROWSY"
                self.mouth_counter = 0
        else:
            self.eye_counter = max(0, self.eye_counter - 1)
        
        # MOUTH DETECTION (Yawning)
        if ear >= self.EYE_AR_THRESH * 0.9:
            if mar > self.MOUTH_AR_THRESH:
                self.mouth_counter += 1
                if self.mouth_counter >= self.CONSEC_FRAMES_MOUTH:
                    if new_status == "DROWSY":
                        new_status = "DROWSY & YAWNING"
                    else:
                        new_status = "YAWNING"
            else:
                self.mouth_counter = max(0, self.mouth_counter - 1)
        else:
            self.mouth_counter = 0
        
        # Blink detection
        if blink_detected and new_status == "NORMAL":
            new_status = "BLINKING"
            # Update blink history
            self.total_blinks += 1
            self.blink_history.append(time.time())
        
        return new_status, blink_detected
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection - UPDATED"""
        self.frame_count += 1
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Mirror frame for natural viewing
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_processor.process(rgb)
        
        # Default values
        ear = 0.0
        mar = 0.0
        eye_distance = 0.0
        eye_asymmetry = 0.0
        blink_detected = False
        self.face_detected = False
        face_bbox = None
        
        # Variables for ML prediction
        left_eye_points = []
        right_eye_points = []
        mouth_points = []
        
        if results.multi_face_landmarks:
            self.face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Draw face bounding box - STORE COORDINATES
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]
            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                padding = 20
                x_min = max(0, x_min - padding)
                x_max = min(w, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(h, y_max + padding)
                face_bbox = (x_min, y_min, x_max, y_max)
            
            # Extract facial features for ML model
            (left_eye_points, right_eye_points, mouth_points, 
             ear, mar, eye_distance, eye_asymmetry) = self.face_processor.extract_facial_features(
                face_landmarks, frame.shape
            )
            
            # Calculate eye asymmetry from EAR values
            ear_left = self.calculate_ear_from_points(left_eye_points)
            ear_right = self.calculate_ear_from_points(right_eye_points)
            eye_asymmetry = abs(ear_left - ear_right)
            
            # Draw landmarks for visualization
            frame = self.face_processor.draw_landmarks(
                frame, left_eye_points, right_eye_points, mouth_points
            )
            
            # Add to history for graphs
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            
            # Use ML model for prediction if available
            if self.ml_model.is_available:
                self.ml_accuracy, self.ml_confidence = self.ml_model.predict_drowsiness(
                    left_eye_points, right_eye_points, mouth_points,
                    ear, mar, eye_distance, eye_asymmetry
                )
            
            # Detect state
            status, blink_detected = self.detect_state(ear, mar, self.prev_ear)
            
            # Log status changes
            if status != self.status and status != "BLINKING":
                if status != "NORMAL":
                    print(f"\n🚨 Status changed: {self.status} → {status}")
                    print(f"   📊 EAR: {ear:.3f}, MAR: {mar:.3f}")
                    if self.ml_model.is_available:
                        print(f"   🤖 ML Accuracy: {self.ml_accuracy:.1%} (Confidence: {self.ml_confidence:.1%})")
                self.status = status
            
            # Add status to history
            self.status_history.append(self.status)
            
            # Store previous EAR for blink detection
            self.prev_ear = ear
        else:
            # No face detected
            self.status = "NO DRIVER"
            ear = 0.0
            mar = 0.0
            
            # Add placeholder values to history
            if len(self.ear_history) > 0:
                self.ear_history.append(self.ear_history[-1])
                self.mar_history.append(self.mar_history[-1])
            else:
                self.ear_history.append(0.0)
                self.mar_history.append(0.0)
        
        # ============================================
        # DRIVER RECOGNITION - UPDATED AND FIXED
        # ============================================
        if self.face_detected:
            # Use the global driver_classifier from dashboard module
            if hasattr(driver_classifier, 'recognize_driver'):
                try:
                    # Recognize driver
                    recognized_name, confidence = driver_classifier.recognize_driver(frame)
                    
                    # Log recognition occasionally
                    if self.frame_count % 60 == 0:  # Every 60 frames
                        print(f"👤 Driver Recognition: {recognized_name} ({confidence:.1%} confidence)")
                    
                    # Update driver info
                    self.driver_name = recognized_name
                    self.driver_confidence = confidence
                    
                    # Store in classifier for dashboard access
                    if hasattr(driver_classifier, 'current_driver_name'):
                        driver_classifier.current_driver_name = recognized_name
                        driver_classifier.current_confidence = confidence
                    
                except Exception as e:
                    print(f"⚠️ Driver recognition error: {e}")
                    self.driver_name = "Unknown (Error)"
                    self.driver_confidence = 0.0
            else:
                # Driver recognition not available
                if self.frame_count % 120 == 0:  # Every 120 frames
                    print("⚠️ Driver recognition function not available")
                self.driver_name = "Unknown (No Classifier)"
                self.driver_confidence = 0.0
        else:
            # No face detected
            self.driver_name = "No Face Detected"
            self.driver_confidence = 0.0
        
        # ============================================
        # UPDATE DASHBOARD DATA
        # ============================================
        self.dashboard_data.update({
            'ear': ear,
            'mar': mar,
            'status': self.status,
            'blinks': self.total_blinks,
            'fps': self.fps,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'face_detected': self.face_detected,
            'ml_enabled': self.ml_model.is_available,
            'ml_accuracy': self.ml_accuracy,
            'ml_confidence': self.ml_confidence,
            'driver_name': self.driver_name,
            'driver_confidence': self.driver_confidence
        })
        
        # Create dashboards
        dashboard1 = create_dashboard1(frame, ear, mar, self.status, blink_detected, 
                                      self.face_detected, self.driver_name, self.fps,
                                      self.total_blinks, self.current_blink_rate,
                                      self.ml_model, face_bbox)
        
        dashboard2 = create_dashboard2(self.dashboard_data, self.ear_history,
                                     self.mar_history, self.ml_model)
        
        return dashboard1, dashboard2, ear, mar, self.status
    
    def run(self):
        """Main execution loop"""
        print("\n" + "="*80)
        print("🚀 STARTING DRIVER DROWSINESS MONITORING")
        print("="*80)
        print("📊 Dashboard 1: Live driver monitoring")
        print("📈 Dashboard 2: Analytics dashboard with graphs")
        print(f"📷 Active Camera: {'External' if self.camera_index > 0 else 'Built-in'} (Index: {self.camera_index})")
        print(f"🤖 Drowsiness ML Model: {'✅ Enabled' if self.ml_model.is_available else '❌ Disabled'}")
        if self.ml_model.is_available:
            print(f"   • Features: {self.ml_model.num_features}")
        
        # Driver recognition status
        self.print_driver_classifier_status()
        
        print("\n🎮 KEYBOARD CONTROLS:")
        print("  • Q = Quit detection")
        print("  • R = Reset counters")
        print("  • P = Pause/Resume")
        print("  • S = Show driver recognition info")
        print("  • D = Toggle driver recognition debug")
        print("="*80 + "\n")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {self.camera_index}")
            print("🔧 Trying built-in camera...")
            self.camera_index = 0
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Error: Could not open any camera")
                return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        paused = False
        show_driver_info = False
        debug_driver_recognition = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"❌ Camera {self.camera_index} error")
                        break
                    
                    # Process frame
                    dashboard1, dashboard2, ear, mar, status = self.process_frame(frame)
                    
                    # Display dashboards
                    cv2.imshow('DASHBOARD 1: Live Monitoring', dashboard1)
                    cv2.imshow('DASHBOARD 2: Analytics Dashboard', dashboard2)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n🛑 Stopping monitoring...")
                    break
                elif key == ord('r'):
                    self.eye_counter = 0
                    self.mouth_counter = 0
                    self.total_blinks = 0
                    self.status = "NORMAL"
                    self.ear_history.clear()
                    self.mar_history.clear()
                    self.ml_accuracy = 0.0
                    self.ml_confidence = 0.0
                    self.driver_name = "Unknown"
                    self.driver_confidence = 0.0
                    print("✅ Counters and history reset")
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️  {'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    show_driver_info = not show_driver_info
                    if show_driver_info:
                        print("\n👤 DRIVER RECOGNITION INFORMATION")
                        print("-" * 40)
                        print(f"Current Driver: {self.driver_name}")
                        print(f"Confidence: {self.driver_confidence:.1%}")
                        print(f"Frame Count: {self.frame_count}")
                        print(f"Face Detected: {self.face_detected}")
                        
                        if hasattr(driver_classifier, 'model_loaded'):
                            print(f"Classifier Loaded: {driver_classifier.model_loaded}")
                        if hasattr(driver_classifier, 'driver_names'):
                            print(f"Registered Drivers: {len(driver_classifier.driver_names)}")
                            if len(driver_classifier.driver_names) > 0:
                                print("Driver Database:")
                                for driver_id, name in driver_classifier.driver_names.items():
                                    print(f"  • {name} (ID: {driver_id})")
                        print("-" * 40)
                elif key == ord('d'):
                    debug_driver_recognition = not debug_driver_recognition
                    if hasattr(driver_classifier, 'recognition_interval'):
                        if debug_driver_recognition:
                            driver_classifier.recognition_interval = 1  # 1 second for debugging
                            print("🔍 Driver recognition debug ON (recognizing every second)")
                        else:
                            driver_classifier.recognition_interval = 5  # Back to 5 seconds
                            print("🔍 Driver recognition debug OFF")
                elif key == 27:  # ESC key
                    print("\n🛑 Stopping monitoring (ESC pressed)...")
                    break
        
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring (Keyboard Interrupt)...")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.print_session_summary()
    
    def print_session_summary(self):
        """Print session summary"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"⏱️  Duration: {total_time:.1f} seconds")
        print(f"📈 Frames processed: {self.frame_count}")
        print(f"⚡ Average FPS: {self.fps:.1f}")
        print(f"👁️  Total Blinks: {self.total_blinks}")
        print(f"📊 Avg Blink Rate: {self.current_blink_rate:.1f}/min")
        
        # Driver info
        print(f"\n👤 DRIVER INFORMATION:")
        print(f"   • Final Driver: {self.driver_name}")
        print(f"   • Confidence: {self.driver_confidence:.1%}")
        
        if hasattr(driver_classifier, 'model_loaded'):
            if driver_classifier.model_loaded:
                print(f"   • Classifier Status: ✅ Active")
                if hasattr(driver_classifier, 'recognition_count'):
                    print(f"   • Recognition Attempts: {driver_classifier.recognition_count}")
            else:
                print(f"   • Classifier Status: ❌ Inactive (Training required)")
        
        print(f"\n📷 Camera Used: {'External' if self.camera_index > 0 else 'Built-in'}")
        print(f"🤖 Drowsiness ML Model: {'Active' if self.ml_model.is_available else 'Inactive'}")
        
        if self.ml_model.is_available and self.ml_accuracy > 0:
            print(f"🤖 Final ML Accuracy: {self.ml_accuracy:.1%} (Confidence: {self.ml_confidence:.1%})")
        
        # Status summary
        if self.face_detected and "DROWSY" in self.status:
            print(f"\n⚠️  FINAL STATUS: {self.status}")
            print("   ⚠️  WARNING: Driver was drowsy during monitoring!")
        elif self.face_detected and "YAWNING" in self.status:
            print(f"\n⚠️  FINAL STATUS: {self.status}")
            print("   ⚠️  WARNING: Driver was yawning during monitoring!")
        elif self.face_detected:
            print(f"\n✅ FINAL STATUS: {self.status}")
            print("   ✅ Driver was alert during monitoring")
        else:
            print(f"\n❌ FINAL STATUS: {self.status}")
            print("   ❌ No driver detected")
        
        print("="*60)
        print("✅ Session completed successfully!\n")