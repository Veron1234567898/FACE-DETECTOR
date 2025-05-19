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

# Import the object detector
from object_detector import ObjectInFrontDetector

class FaceDetector:
    """
    Face detector class supporting both MediaPipe and OpenCV's built-in face detector.
    Also includes occlusion detection to identify objects covering the face.
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
        
        # Initialize tracking variables
        self.prev_faces = []
        self.tracking_enabled = True  # Enable tracking between frames
        self.track_quality_threshold = 0.3  # Minimum quality for tracking (0-1)
        self.max_tracking_age = 5  # Maximum frames to track without detection
        self.tracking_ages = []  # How many frames each face has been tracked
        
        # Initialize object-in-front detector
        self.occlusion_detector = ObjectInFrontDetector()
        self.detect_occlusions = True  # Flag to enable/disable object detection
        
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
            self.profile_model_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
            
            # Check if the model file exists, if not, use the built-in OpenCV model
            if not os.path.exists(model_path):
                model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.alt_model_path = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
            
            # Load the face detector - optimize parameters for speed
            self.face_cascade = cv2.CascadeClassifier(model_path)
            # Load alternative model as backup
            self.alt_face_cascade = cv2.CascadeClassifier(self.alt_model_path)
            # Load profile face detector for side views
            self.profile_cascade = cv2.CascadeClassifier(self.profile_model_path)
            
            # For eye detection (useful for landmarks)
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            
            # Store last frame for tracking
            self.prev_gray = None
            
        # Define optimized detection parameters
        self.min_neighbors = 3  # Lower value = more detections but more false positives
        self.scale_factor = 1.2  # Higher value = faster but might miss faces
        
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _update_tracking(self, gray_frame, detected_faces):
        """Update face tracking using optical flow or simple IOU matching."""
        if not self.tracking_enabled or not self.prev_faces:
            self.prev_faces = detected_faces
            self.tracking_ages = [0] * len(detected_faces)
            if gray_frame is not None:
                self.prev_gray = gray_frame.copy()
            return detected_faces
        
        # If we have detections, match with previous faces
        if detected_faces:
            # Simple IOU-based tracking
            matched_indices = [-1] * len(self.prev_faces)
            new_faces = []
            new_ages = []
            
            # Match each previous face to a current detection
            for i, prev_face in enumerate(self.prev_faces):
                best_iou = self.track_quality_threshold
                best_match = -1
                
                for j, curr_face in enumerate(detected_faces):
                    iou = self._calculate_iou(prev_face['bbox'], curr_face['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                
                if best_match >= 0:
                    # Found a match - update tracking age
                    matched_indices[i] = best_match
                    new_face = detected_faces[best_match].copy()
                    new_ages.append(0)  # Reset age for matched face
                    new_faces.append(new_face)
                elif self.tracking_ages[i] < self.max_tracking_age:
                    # No match but still tracking - increment age
                    self.tracking_ages[i] += 1
                    new_ages.append(self.tracking_ages[i])
                    new_faces.append(prev_face)
            
            # Add unmatched new detections
            for j, curr_face in enumerate(detected_faces):
                if j not in matched_indices:
                    new_faces.append(curr_face)
                    new_ages.append(0)
            
            self.prev_faces = new_faces
            self.tracking_ages = new_ages
            if gray_frame is not None:
                self.prev_gray = gray_frame.copy()
                
            return new_faces
        else:
            # No detections, update tracking ages
            new_faces = []
            new_ages = []
            
            for i, face in enumerate(self.prev_faces):
                if self.tracking_ages[i] < self.max_tracking_age:
                    self.tracking_ages[i] += 1
                    new_ages.append(self.tracking_ages[i])
                    new_faces.append(face)
            
            self.prev_faces = new_faces
            self.tracking_ages = new_ages
            return new_faces
    
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
        
        # Prepare grayscale image for both detection and tracking
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply a slight Gaussian blur to reduce noise and improve detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
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
                            
                            # Detect objects in front of face if enabled
                            if self.detect_occlusions:
                                objects = self.occlusion_detector.detect_objects(face_roi, (x_min, y_min, x_max, y_max))
                                face_info['occlusions'] = objects
                                print(f"DEBUG: Detected {len(objects)} objects in MediaPipe path")
                                
                            detected_faces.append(face_info)
        else:
            # Using OpenCV's face detector with optimized parameters
            # Detect faces using Haar cascades with optimized parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=self.scale_factor, 
                minNeighbors=self.min_neighbors, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no faces found with the primary detector, try alternative detector
            if len(faces) == 0:
                faces = self.alt_face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=self.scale_factor, 
                    minNeighbors=self.min_neighbors, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            # If still no faces, try profile face detection
            if len(faces) == 0:
                # Left profile
                faces = self.profile_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=self.scale_factor, 
                    minNeighbors=self.min_neighbors, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Right profile (flip the image)
                if len(faces) == 0:
                    flipped = cv2.flip(gray, 1)  # 1 for horizontal flip
                    faces_flipped = self.profile_cascade.detectMultiScale(
                        flipped, 
                        scaleFactor=self.scale_factor, 
                        minNeighbors=self.min_neighbors, 
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    # Convert flipped coordinates back to original image
                    for (x, y, width, height) in faces_flipped:
                        faces = np.vstack((faces, np.array([gray.shape[1] - x - width, y, width, height]))) if len(faces) > 0 else np.array([[gray.shape[1] - x - width, y, width, height]])
            
            # Process detected faces
            for (x, y, width, height) in faces:
                x_min, y_min = x, y
                x_max, y_max = x + width, y + height
                
                # Create a dict with detection info
                face_info = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'confidence': 0.9,  # Assign a reasonable confidence value
                    'landmarks': {}
                }
                
                # Extract face region
                face_roi = gray[y_min:y_max, x_min:x_max] if 0 <= y_min < y_max < gray.shape[0] and 0 <= x_min < x_max < gray.shape[1] else None
                
                if face_roi is not None and face_roi.size > 0:
                    # Try to detect eyes in the face region to use as landmarks
                    eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
                    
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
                    else:
                        # Estimate eye positions if detection failed
                        face_info['landmarks']['left_eye'] = (x_min + width // 4, y_min + height // 3)
                        face_info['landmarks']['right_eye'] = (x_min + 3 * width // 4, y_min + height // 3)
                    
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
                face_color_roi = image[y_min:y_max, x_min:x_max] if 0 <= y_min < y_max < image.shape[0] and 0 <= x_min < x_max < image.shape[1] else None
                if face_color_roi is not None and face_color_roi.size > 0:  # Check if ROI is not empty
                    face_info['face_roi'] = face_color_roi
                    
                    # Detect objects in front of face if enabled
                    if self.detect_occlusions:
                        objects = self.occlusion_detector.detect_objects(face_color_roi, (x_min, y_min, x_max, y_max))
                        face_info['occlusions'] = objects
                        print(f"DEBUG: Detected {len(objects)} objects in OpenCV path")
                    
                    detected_faces.append(face_info)
        
        # Apply face tracking for smoother results
        tracked_faces = self._update_tracking(gray, detected_faces)
                
        return tracked_faces
    
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
                
            # Draw occlusions if present
            if 'occlusions' in face and face['occlusions']:
                for occlusion in face['occlusions']:
                    # Draw red box around occlusion
                    o_x_min, o_y_min, o_x_max, o_y_max = occlusion['bbox']
                    cv2.rectangle(
                        annotated_image,
                        (int(o_x_min), int(o_y_min)),
                        (int(o_x_max), int(o_y_max)),
                        (0, 0, 255),  # Red color for occlusions
                        2
                    )
                    
                    # Add label for the occluded region
                    label = f"{occlusion['region']}"
                    cv2.putText(
                        annotated_image,
                        label,
                        (int(o_x_min), int(o_y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
        
        return annotated_image
