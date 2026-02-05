# face_processor.py - UPDATED FOR 56 FEATURES
import cv2
import numpy as np
import mediapipe as mp

class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for 56-feature model (matching your training)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [13, 14, 78, 308, 17, 18]
        
        # Additional landmarks for 56 features
        self.FACE_CONTOUR = [10, 338, 297, 332]  # Face boundaries
        self.EYEBROW = [70, 63, 105, 66]  # Eyebrow movement
    
    def process(self, rgb_image):
        """Process RGB image to detect faces"""
        return self.face_mesh.process(rgb_image)
    
    def extract_facial_features(self, face_landmarks, image_shape):
        """
        Extract ALL 56 facial features for ML model
        Returns normalized coordinates (0-1 range) exactly like training data
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image_shape: (height, width) of image
        
        Returns:
            tuple: (left_eye_points, right_eye_points, mouth_points, ear, mar, eye_distance)
        """
        # Extract normalized coordinates (0-1 range) - IMPORTANT for ML model!
        left_eye_points = []
        right_eye_points = []
        mouth_points = []
        face_contour_points = []
        eyebrow_points = []
        
        # Left eye landmarks (6 points = 12 features)
        for idx in self.LEFT_EYE:
            lm = face_landmarks.landmark[idx]
            left_eye_points.append((lm.x, lm.y))
        
        # Right eye landmarks (6 points = 12 features)
        for idx in self.RIGHT_EYE:
            lm = face_landmarks.landmark[idx]
            right_eye_points.append((lm.x, lm.y))
        
        # Mouth landmarks (6 points = 12 features)
        for idx in self.MOUTH:
            lm = face_landmarks.landmark[idx]
            mouth_points.append((lm.x, lm.y))
        
        # Face contour landmarks (4 points = 8 features)
        for idx in self.FACE_CONTOUR:
            lm = face_landmarks.landmark[idx]
            face_contour_points.append((lm.x, lm.y))
        
        # Eyebrow landmarks (4 points = 8 features)
        for idx in self.EYEBROW:
            lm = face_landmarks.landmark[idx]
            eyebrow_points.append((lm.x, lm.y))
        
        # Calculate EAR using normalized coordinates
        left_ear = self.calculate_ear(left_eye_points)
        right_ear = self.calculate_ear(right_eye_points)
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR using normalized coordinates
        mar = self.calculate_mar(mouth_points)
        
        # Calculate eye distance (in normalized coordinates)
        left_eye_center = np.mean(left_eye_points, axis=0)
        right_eye_center = np.mean(right_eye_points, axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        # Calculate eye asymmetry (new feature)
        eye_asymmetry = abs(left_ear - right_ear)
        
        # Return all the extracted features
        return (left_eye_points, right_eye_points, mouth_points, 
                ear, mar, eye_distance, eye_asymmetry)
    
    def extract_facial_features_for_ml(self, face_landmarks):
        """
        Alias for extract_facial_features - for backward compatibility
        """
        return self.extract_facial_features(face_landmarks, (480, 640))
    
    def calculate_ear_mar_for_display(self, face_landmarks, image_height, image_width):
        """
        Calculate EAR and MAR for display purposes (uses pixel coordinates)
        """
        # Extract eye and mouth points in pixel coordinates
        left_eye = self.extract_landmarks_pixels(face_landmarks, self.LEFT_EYE, image_width, image_height)
        right_eye = self.extract_landmarks_pixels(face_landmarks, self.RIGHT_EYE, image_width, image_height)
        mouth = self.extract_landmarks_pixels(face_landmarks, self.MOUTH, image_width, image_height)
        
        # Calculate EAR
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Calculate MAR
        mar = self.calculate_mar(mouth)
        
        return ear, mar
    
    def extract_landmarks_pixels(self, face_landmarks, indices, img_w, img_h):
        """Extract specific landmarks as pixel coordinates (for display)"""
        points = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            points.append((x, y))
        return points
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
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
    
    def calculate_mar(self, mouth_points):
        """Calculate Mouth Aspect Ratio"""
        if len(mouth_points) < 6:
            return 0.3
        
        top_left = mouth_points[0]
        top_right = mouth_points[1]
        left_corner = mouth_points[2]
        right_corner = mouth_points[3]
        bottom_left = mouth_points[4]
        bottom_right = mouth_points[5]
        
        top_center = ((top_left[0] + top_right[0]) / 2, 
                     (top_left[1] + top_right[1]) / 2)
        bottom_center = ((bottom_left[0] + bottom_right[0]) / 2,
                        (bottom_left[1] + bottom_right[1]) / 2)
        
        vertical = np.linalg.norm(np.array(top_center) - np.array(bottom_center))
        horizontal = np.linalg.norm(np.array(left_corner) - np.array(right_corner))
        
        if horizontal == 0:
            return 0.3
        
        return vertical / horizontal
    
    def draw_landmarks(self, frame, left_eye_points, right_eye_points, mouth_points):
        """Draw facial landmarks on frame for visualization"""
        h, w = frame.shape[:2]
        
        # Draw left eye points
        for point in left_eye_points:
            x, y = int(point[0] * w), int(point[1] * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw right eye points
        for point in right_eye_points:
            x, y = int(point[0] * w), int(point[1] * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Draw mouth points
        for point in mouth_points:
            x, y = int(point[0] * w), int(point[1] * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
        return frame