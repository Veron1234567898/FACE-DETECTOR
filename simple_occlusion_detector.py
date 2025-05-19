import cv2
import numpy as np

class SimpleOcclusionDetector:
    """
    A simplified occlusion detector that directly identifies objects covering faces
    using basic image processing techniques.
    """
    def __init__(self):
        """Initialize the simple occlusion detector."""
        # Define occlusion detection sensitivity - more conservative to avoid false positives
        self.motion_threshold = 50  # Higher threshold for motion detection (less sensitive)
        self.edge_threshold = 60  # Higher threshold for edge detection (less sensitive)
        self.min_contour_area = 800  # Larger minimum area to avoid detecting small facial features
        
        # Store the previous face image for comparison
        self.prev_face = None
        self.face_features_mask = None  # Will store normal facial features to exclude
        self.detection_countdown = 0  # Wait a few frames before starting detection
        
    def detect_occlusions(self, face_roi, face_bbox):
        """
        Directly detect objects covering parts of the face.
        
        Args:
            face_roi: Face image region
            face_bbox: Face bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            List of detected occlusions with bounding boxes
        """
        if face_roi is None or face_roi.size == 0:
            return []
            
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Find edges using Canny edge detector
        edges = cv2.Canny(filtered, self.edge_threshold, self.edge_threshold * 2)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours in the edges image
        try:
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = list(contours)  # Convert to list to ensure we can extend it later
        except Exception as e:
            print(f"Edge contour detection error: {e}")
            contours = []  # Start with an empty list if edge detection fails
        
        # Get face dimensions
        face_height, face_width = face_roi.shape[:2]
        x_min, y_min, x_max, y_max = face_bbox
        
        # Simple skin color detection in RGB
        skin_mask = None
        if len(face_roi.shape) == 3:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            # Create binary mask for skin
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            # Invert to get potential occlusion areas (non-skin)
            non_skin = cv2.bitwise_not(skin_mask)
            
            # Find contours in non-skin areas
            try:
                non_skin_contours, _ = cv2.findContours(non_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Add these contours to our list if they exist
                if non_skin_contours is not None:
                    contours = list(contours)  # Make sure contours is a list that can be extended
                    contours.extend(non_skin_contours)
            except Exception as e:
                print(f"Non-skin contour detection error: {e}")
                # Continue with the existing contours
        
        # Create direct occlusion check for common occlusion scenarios
        direct_occlusions = []
        
        # Skip a few frames before detecting to avoid false positives when the face first appears
        if self.detection_countdown > 0:
            self.detection_countdown -= 1
            # Store current face for comparison but don't detect yet
            self.prev_face = gray.copy()
            return []
            
        # If no previous face, initialize and wait a few frames
        if self.prev_face is None or self.prev_face.shape != gray.shape:
            self.prev_face = gray.copy()
            self.detection_countdown = 3  # Wait 3 frames before detection starts
            return []
        
        # Calculate absolute difference between current and previous frame
        frame_diff = cv2.absdiff(gray, self.prev_face)
        
        # Apply a stronger blur to remove noise
        frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
        
        # Threshold the difference image with higher threshold
        _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours in the motion mask
        try:
            motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Add these to our contours list for processing if they exist
            if motion_contours is not None:
                contours = list(contours)  # Ensure contours is a list
                
                # Only keep motion contours that are significant in size
                significant_motion_contours = []
                for contour in motion_contours:
                    if cv2.contourArea(contour) > self.min_contour_area * 2:  # Higher threshold for motion
                        significant_motion_contours.append(contour)
                        
                contours.extend(significant_motion_contours)
        except Exception as e:
            print(f"Motion contour detection error: {e}")
            # Continue with existing contours
        
        # Store current face for next comparison
        self.prev_face = gray.copy()
        
        # Create a face features mask to ignore normal facial features
        # This helps avoid false detections of eyes, nose, mouth as occlusions
        if self.face_features_mask is None or self.face_features_mask.shape != gray.shape:
            self.face_features_mask = np.zeros_like(gray)
            
            # Define regions to ignore (typical locations of eyes, nose, mouth)
            # Eyes region
            eye_y_start = int(face_height * 0.2)
            eye_y_end = int(face_height * 0.4)
            left_eye_x = int(face_width * 0.3)
            right_eye_x = int(face_width * 0.7)
            eye_width = int(face_width * 0.15)
            cv2.rectangle(self.face_features_mask, 
                         (left_eye_x - eye_width//2, eye_y_start),
                         (left_eye_x + eye_width//2, eye_y_end), 255, -1)
            cv2.rectangle(self.face_features_mask, 
                         (right_eye_x - eye_width//2, eye_y_start),
                         (right_eye_x + eye_width//2, eye_y_end), 255, -1)
            
            # Nose region
            nose_y_start = int(face_height * 0.4)
            nose_y_end = int(face_height * 0.6)
            nose_x = int(face_width * 0.5)
            nose_width = int(face_width * 0.2)
            cv2.rectangle(self.face_features_mask, 
                         (nose_x - nose_width//2, nose_y_start),
                         (nose_x + nose_width//2, nose_y_end), 255, -1)
            
            # Mouth region
            mouth_y_start = int(face_height * 0.6)
            mouth_y_end = int(face_height * 0.75)
            mouth_x = int(face_width * 0.5)
            mouth_width = int(face_width * 0.3)
            cv2.rectangle(self.face_features_mask, 
                         (mouth_x - mouth_width//2, mouth_y_start),
                         (mouth_x + mouth_width//2, mouth_y_end), 255, -1)
        
        # Process all contours to find potential occlusions
        for contour in contours:
            # Filter out small contours
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate relative size compared to face
            relative_size = (w * h) / (face_width * face_height)
            
            # Skip if too small or too large
            if relative_size < 0.1 or relative_size > 0.6:  # More conservative range
                continue
                
            # IMPORTANT: Skip this contour if it significantly overlaps with known facial features
            # Create a mask from the contour
            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [contour], 0, 255, -1)
            
            # Check overlap with face features mask
            overlap = cv2.bitwise_and(contour_mask, self.face_features_mask)
            overlap_percentage = cv2.countNonZero(overlap) / cv2.countNonZero(contour_mask) if cv2.countNonZero(contour_mask) > 0 else 0
            
            # Skip if the contour is mostly a facial feature (>50% overlap)
            if overlap_percentage > 0.5:
                continue
            
            # Calculate region of the face this is in
            region_y = y / face_height
            
            # Determine region name
            if region_y < 0.3:
                region_name = 'OBJECT over forehead/eyes'
            elif region_y < 0.6:
                region_name = 'OBJECT over nose'
            else:
                region_name = 'OBJECT over mouth/chin'
            
            # Convert contour bbox to global coordinates
            occlusion_box = (
                x_min + x,
                y_min + y,
                x_min + x + w,
                y_min + y + h
            )
            
            # Direct detection: Add this potential occlusion
            direct_occlusions.append({
                'bbox': occlusion_box,
                'region': region_name,
                'confidence': min(1.0, area / 1000)  # Scale confidence by area
            })
        
        return direct_occlusions
    
    def draw_occlusions(self, image, occlusions):
        """
        Draw red boxes around detected occlusions.
        
        Args:
            image: Original image
            occlusions: List of detected occlusions
            
        Returns:
            Image with occlusions highlighted
        """
        # Create a copy of the image
        annotated_image = image.copy()
        
        # Draw each occlusion
        for occlusion in occlusions:
            # Get bounding box
            x_min, y_min, x_max, y_max = occlusion['bbox']
            
            # Draw a red box around the occlusion
            cv2.rectangle(
                annotated_image,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),  # Red color
                3  # Thickness
            )
            
            # Add a semi-transparent overlay
            overlay = annotated_image.copy()
            cv2.rectangle(
                overlay,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),
                -1  # Fill
            )
            
            # Apply overlay with transparency
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
            
            # Add text label
            label = f"Occlusion: {occlusion['region']}"
            
            # Background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated_image,
                (int(x_min), int(y_min) - text_size[1] - 5),
                (int(x_min) + text_size[0], int(y_min)),
                (0, 0, 0),
                -1
            )
            
            # Add text
            cv2.putText(
                annotated_image,
                label,
                (int(x_min), int(y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_image
