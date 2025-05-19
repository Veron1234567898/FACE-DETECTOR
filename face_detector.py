import cv2
import numpy as np
import os

# Try to import mediapipe, but continue if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available. Using OpenCV's built-in face detector instead.")

class FaceDetector:
    """
    Face detector class supporting both MediaPipe and OpenCV's built-in face detector.
    """
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Initialize the face detector.
        
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.
            model_selection: 0 or 1. 0 for short-range detection (2 meters), 1 for full-range detection (5 meters).
        """
        self.min_confidence = min_detection_confidence
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        if MEDIAPIPE_AVAILABLE:
            # Use MediaPipe face detection
            self.detection_method = "mediapipe"
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence,
                model_selection=model_selection
            )
        else:
            # Use OpenCV's built-in face detector
            self.detection_method = "opencv"
            
            # Path to pre-trained face detection model files
            model_path = os.path.join("models", "haarcascade_frontalface_default.xml")
            self.alt_model_path = os.path.join("models", "haarcascade_frontalface_alt.xml")
            
            # Check if the model file exists, if not, use the built-in OpenCV model
            if not os.path.exists(model_path):
                model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.alt_model_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            
            # Load the face detector
            self.face_cascade = cv2.CascadeClassifier(model_path)
            # Load alternative model as backup
            self.alt_face_cascade = cv2.CascadeClassifier(self.alt_model_path)
            
            # For eye detection (useful for landmarks)
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        
    def detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: RGB image
            
        Returns:
            List of detected faces with their bounding boxes and landmarks
        """
        # Initialize an empty list for detected faces
        detected_faces = []
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        if self.detection_method == "mediapipe":
            # Convert to RGB if needed for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
            
            # Process the image
            results = self.face_detection.process(image_rgb)
            
            # Check if any faces were detected
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to absolute
                    x_min = max(0, int(bbox.xmin * w))
                    y_min = max(0, int(bbox.ymin * h))
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure the bounding box doesn't exceed image dimensions
                    x_max = min(w, x_min + width)
                    y_max = min(h, y_min + height)
                    
                    # Calculate the confidence score
                    score = detection.score[0]
                    
                    # Only include faces above the confidence threshold
                    if score >= self.min_confidence:
                        # Create a dict with detection info
                        face_info = {
                            'bbox': (x_min, y_min, x_max, y_max),
                            'confidence': score,
                            'landmarks': {}
                        }
                        
                        # Get facial landmarks (keypoints)
                        for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                            # Convert relative keypoint coordinates to absolute
                            kp_x = int(landmark.x * w)
                            kp_y = int(landmark.y * h)
                            
                            # Map the keypoint index to its name
                            keypoint_names = ['right_eye', 'left_eye', 'nose_tip', 'mouth_center', 'right_ear_tragion', 'left_ear_tragion']
                            if idx < len(keypoint_names):
                                face_info['landmarks'][keypoint_names[idx]] = (kp_x, kp_y)
                        
                        # Extract face region for emotion and age estimation
                        face_roi = image[y_min:y_max, x_min:x_max]
                        if face_roi.size > 0:  # Check if ROI is not empty
                            face_info['face_roi'] = face_roi
                            detected_faces.append(face_info)
        else:
            # Using OpenCV's face detector
            # Convert to grayscale for Haar cascades
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image
            
            # Detect faces using Haar cascades
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # If no faces found with the primary detector, try alternative detector
            if len(faces) == 0:
                faces = self.alt_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, width, height) in faces:
                x_min, y_min = x, y
                x_max, y_max = x + width, y + height
                
                # Create a dict with detection info
                face_info = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': 1.0,  # OpenCV doesn't provide confidence score, so use default
                    'landmarks': {}
                }
                
                # Extract face region
                face_roi = gray[y_min:y_max, x_min:x_max]
                
                # Try to detect eyes in the face region to use as landmarks
                eyes = self.eye_cascade.detectMultiScale(face_roi)
                
                if len(eyes) >= 2:
                    # Sort by x-coordinate to get left and right eye
                    eyes = sorted(eyes, key=lambda eye: eye[0])
                    
                    # Get left and right eye centers
                    left_eye = eyes[0]
                    right_eye = eyes[1]
                    
                    face_info['landmarks']['left_eye'] = (x_min + left_eye[0] + left_eye[2]//2, 
                                                         y_min + left_eye[1] + left_eye[3]//2)
                    face_info['landmarks']['right_eye'] = (x_min + right_eye[0] + right_eye[2]//2, 
                                                          y_min + right_eye[1] + right_eye[3]//2)
                    
                    # Estimate other facial landmarks
                    # Nose tip (estimate at center of face, 60% from top)
                    nose_x = (x_min + x_max) // 2
                    nose_y = y_min + int(height * 0.6)
                    face_info['landmarks']['nose_tip'] = (nose_x, nose_y)
                    
                    # Mouth center (estimate at center of face, 80% from top)
                    mouth_x = (x_min + x_max) // 2
                    mouth_y = y_min + int(height * 0.8)
                    face_info['landmarks']['mouth_center'] = (mouth_x, mouth_y)
                
                # Extract face region for emotion and age estimation (use color for better model performance)
                face_color_roi = image[y_min:y_max, x_min:x_max]
                if face_color_roi.size > 0:  # Check if ROI is not empty
                    face_info['face_roi'] = face_color_roi
                    detected_faces.append(face_info)
                
        return detected_faces
    
    def draw_detections(self, image, detections, show_confidence=True, box_color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes and landmarks on the image.
        
        Args:
            image: Original BGR image
            detections: List of face detections from detect_faces()
            show_confidence: Whether to display detection confidence
            box_color: RGB color for bounding box
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        annotated_image = image.copy()
        
        for face in detections:
            # Draw bounding box
            x_min, y_min, x_max, y_max = face['bbox']
            cv2.rectangle(
                annotated_image, 
                (x_min, y_min), (x_max, y_max),
                box_color, 
                thickness
            )
            
            # Optionally draw confidence score
            if show_confidence:
                conf_text = f"{face['confidence']:.2f}"
                cv2.putText(
                    annotated_image, 
                    conf_text, 
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    box_color, 
                    thickness
                )
            
            # Draw landmarks
            for landmark_name, (x, y) in face['landmarks'].items():
                cv2.circle(annotated_image, (x, y), 2, (255, 0, 0), -1)
        
        return annotated_image
