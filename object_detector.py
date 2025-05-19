import cv2
import numpy as np

class ObjectInFrontDetector:
    """
    Detects objects placed in front of the face by looking for 
    significant changes or non-face objects within the face region.
    """
    
    def __init__(self):
        # Store previous face for comparison
        self.prev_face = None
        self.reference_face = None
        self.reference_count = 0
        
        # Motion threshold (lower = more sensitive)
        self.motion_threshold = 30
        
    def detect_objects(self, face_roi, face_bbox):
        """
        Detect objects in front of the face.
        
        Args:
            face_roi: Face region of interest
            face_bbox: Face bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            List of detected objects
        """
        if face_roi is None or face_roi.size == 0:
            return []
            
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
            
        # If we don't have a reference face yet, start building one
        if self.reference_face is None or self.reference_face.shape != gray.shape:
            self.reference_face = gray.copy()
            self.reference_count = 1
            return []  # Need more frames to establish reference
        
        # Build up reference face over first few frames
        if self.reference_count < 5:
            # Average with existing reference
            alpha = 1.0 / (self.reference_count + 1)
            cv2.addWeighted(gray, alpha, self.reference_face, 1.0 - alpha, 0, self.reference_face)
            self.reference_count += 1
            return []  # Still building reference
        
        # Compare current face with reference to detect objects
        objects = []
        
        # Calculate difference from reference face
        diff = cv2.absdiff(gray, self.reference_face)
        
        # Threshold to find significant changes
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Clean up the threshold image
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours of changed areas
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get dimensions of face ROI
        h, w = gray.shape[:2]
        x_min, y_min, x_max, y_max = face_bbox
        
        # Check if any contour is large enough to be an object
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Only consider larger changes
            if area > 0.05 * h * w:  # At least 5% of face
                # Get bounding box
                x, y, w_c, h_c = cv2.boundingRect(contour)
                
                # Convert to global coordinates
                obj_box = (
                    x_min + x,
                    y_min + y,
                    x_min + x + w_c,
                    y_min + y + h_c
                )
                
                # Add to list of detected objects
                objects.append({
                    'bbox': obj_box,
                    'confidence': min(1.0, area / (h * w)),
                    'type': 'OBJECT DETECTED'
                })
        
        # Also check for overall face obstruction
        if np.mean(thresh) > 30:  # If more than 30% of face is different
            # Entire face might be covered
            objects.append({
                'bbox': face_bbox,
                'confidence': np.mean(thresh) / 255.0,
                'type': 'FACE COVERED'
            })
            
        return objects
    
    def draw_objects(self, image, objects):
        """
        Draw objects on the image.
        
        Args:
            image: Image to draw on
            objects: List of detected objects
            
        Returns:
            Image with objects drawn
        """
        if not objects:
            return image
            
        result = image.copy()
        
        # Add warning banner at top of image
        cv2.rectangle(result, (0, 0), (result.shape[1], 80), (0, 0, 255), -1)
        cv2.putText(
            result,
            "WARNING: OBJECT COVERING FACE",
            (result.shape[1]//2 - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Draw each detected object
        for obj in objects:
            x_min, y_min, x_max, y_max = obj['bbox']
            
            # Draw red box
            cv2.rectangle(
                result,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),  # Red
                3
            )
            
            # Add semi-transparent overlay
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),
                -1
            )
            # Apply overlay with transparency
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
            
            # Add label
            label = f"{obj['type']} ({obj['confidence']:.2f})"
            cv2.putText(
                result,
                label,
                (int(x_min), int(y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return result
