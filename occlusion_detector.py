import cv2
import numpy as np

class OcclusionDetector:
    """
    Detects occlusions on faces and identifies them with bounding boxes.
    Can detect objects like masks, hands, glasses, or other objects covering parts of the face.
    """
    def __init__(self):
        """Initialize the occlusion detector."""
        # Define face landmarks regions for occlusion detection
        # Each region is defined as a relative area of the face where we'll check for occlusion
        self.regions = {
            'forehead': {'y_min': 0.0, 'y_max': 0.2, 'x_min': 0.2, 'x_max': 0.8},
            'eyes': {'y_min': 0.2, 'y_max': 0.4, 'x_min': 0.1, 'x_max': 0.9},
            'nose': {'y_min': 0.4, 'y_max': 0.6, 'x_min': 0.3, 'x_max': 0.7},
            'mouth': {'y_min': 0.6, 'y_max': 0.8, 'x_min': 0.25, 'x_max': 0.75},
            'chin': {'y_min': 0.8, 'y_max': 0.95, 'x_min': 0.3, 'x_max': 0.7}
        }
        
        # Super sensitive detection parameters to ensure we catch any objects covering the face
        self.edge_threshold = 30  # Very low edge threshold to detect more edges
        self.variance_threshold = 200  # Lower variance threshold to detect more texture differences
        self.occlusion_threshold = 0.2  # Very low proportion required (extremely sensitive)
        self.min_contour_area_ratio = 0.01  # Detect very small objects
        self.max_contour_area_ratio = 0.95  # Allow almost the entire face to be covered
        self.confidence_threshold = 0.1  # Very low confidence threshold
        
    def _detect_edges(self, face_roi):
        """Detect edges in the face ROI to identify potential occlusions."""
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
            
        # Apply histogram equalization to enhance contrast
        gray = cv2.equalizeHist(gray)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 2)
        
        # Dilate edges to make them more prominent
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def _compute_texture_variance(self, face_roi):
        """Compute texture variance in the face ROI to identify potential occlusions."""
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
            
        # Enhance image contrast to make occlusions more visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Calculate local variance using a 5x5 window
        local_mean = cv2.blur(enhanced.astype(np.float32), (5, 5))
        local_std = np.sqrt(cv2.blur(np.square(enhanced.astype(np.float32)), (5, 5)) - np.square(local_mean))
        
        # Create two masks - one for low variance (uniform objects like hands)
        # and one for high variance (textured objects like masks, glasses)
        low_var_mask = (local_std < self.variance_threshold / 2).astype(np.uint8) * 255
        high_var_mask = (local_std > self.variance_threshold * 1.5).astype(np.uint8) * 255
        
        # Combine both masks - now we detect both uniform and highly textured objects
        occlusion_mask = cv2.bitwise_or(low_var_mask, high_var_mask)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        
        return occlusion_mask
    
    def _compute_skin_mask(self, face_roi):
        """Compute a skin color mask to identify non-skin areas as potential occlusions."""
        # Convert to HSV color space which is better for skin detection
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color in HSV (broadened range to be more inclusive)
            lower_skin = np.array([0, 15, 60], dtype=np.uint8)  # More inclusive lower bounds
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)  # Expanded hue range
            
            # Create binary mask for skin
            skin_mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Add the second range for skin colors (higher hue values)
            lower_skin2 = np.array([165, 15, 60], dtype=np.uint8)  # More inclusive lower bounds
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            # Combine the two skin masks
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            # Apply morphological operations to clean up the skin mask
            kernel = np.ones((3, 3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Invert the mask to get non-skin areas
            occlusion_mask = cv2.bitwise_not(skin_mask)
            
            # Apply additional processing to enhance potential occlusions
            occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
            
            return occlusion_mask
        else:
            # If not a color image, can't detect skin
            return np.zeros(face_roi.shape, dtype=np.uint8)
    
    def _validate_face(self, face_roi):
        """
        Validate if the ROI actually contains a face to prevent false positives.
        
        Args:
            face_roi: The face region of interest
            
        Returns:
            Boolean indicating if this appears to be a valid face
        """
        if face_roi is None or face_roi.size == 0:
            return False
            
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
            
        # Method 1: Check for typical face histogram distribution (skin tones)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist / hist.sum()  # Normalize histogram
        
        # Faces typically have a more balanced histogram distribution
        # Non-faces (like walls) often have very concentrated values
        hist_std = np.std(hist_normalized)
        if hist_std < 0.005:  # Very uniform histogram = probably not a face (lowered threshold)
            return False
            
        # Method 2: Basic skin tone detection (if color image)
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            # Define range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_percentage = cv2.countNonZero(skin_mask) / (face_roi.shape[0] * face_roi.shape[1])
            
            # Most faces should have a significant percentage of skin tones
            # But we don't want to be too strict because hands/objects covering face can reduce skin percentage
            if skin_percentage < 0.15:  # Less than 15% skin = probably not a face (lowered threshold)
                return False
        
        return True
        
    def detect_occlusions(self, face_roi, face_bbox):
        """
        Detect occlusions on a face ROI.
        
        Args:
            face_roi: The face region of interest
            face_bbox: The face bounding box coordinates (x_min, y_min, x_max, y_max)
            
        Returns:
            List of dictionaries containing information about detected occlusions
        """
        if face_roi is None or face_roi.size == 0:
            return []
            
        # Only validate if not in a face if we have no landmarks
        # When the face is partially covered, we still want to detect the occlusions
        # This way, even heavily occluded faces will still be checked for occlusions
        if not self._validate_face(face_roi):
            # Special check - if the ROI has very high contrast or many edges,
            # it might be an object covering a face, so we still want to detect it
            edges = self._detect_edges(face_roi)
            edge_percentage = cv2.countNonZero(edges) / (face_roi.shape[0] * face_roi.shape[1])
            
            if edge_percentage < 0.1:  # If very few edges, probably not a face or object
                return []
        
        # Get face dimensions
        h, w = face_roi.shape[:2]
        face_area = h * w
        
        # Detect edges in the face - sharp edges can indicate occlusion boundaries
        edges = self._detect_edges(face_roi)
        
        # Compute texture variance - uniform areas might be occlusions
        variance_mask = self._compute_texture_variance(face_roi)
        
        # Compute skin mask - non-skin areas might be occlusions
        skin_mask = self._compute_skin_mask(face_roi)
        
        # Combine the masks
        combined_mask = cv2.bitwise_or(edges, variance_mask)
        combined_mask = cv2.bitwise_or(combined_mask, skin_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get face bbox information
        x_min, y_min, x_max, y_max = face_bbox
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        # Check for occlusions in each defined face region
        occlusions = []
        
        for region_name, region in self.regions.items():
            # Calculate region coordinates
            region_x_min = int(region['x_min'] * w)
            region_y_min = int(region['y_min'] * h)
            region_x_max = int(region['x_max'] * w)
            region_y_max = int(region['y_max'] * h)
            
            # Create a mask for this region
            region_mask = np.zeros_like(combined_mask)
            region_mask[region_y_min:region_y_max, region_x_min:region_x_max] = 255
            
            # Check if any contours intersect with this region
            for contour in contours:
                # Calculate area of the contour
                contour_area = cv2.contourArea(contour)
                contour_area_ratio = contour_area / face_area
                
                # Skip if contour is too small or too large relative to face
                if contour_area_ratio < self.min_contour_area_ratio or contour_area_ratio > self.max_contour_area_ratio:
                    continue
                    
                # Create a mask from the contour
                contour_mask = np.zeros_like(combined_mask)
                cv2.drawContours(contour_mask, [contour], 0, 255, -1)
                
                # Calculate area of intersection between contour and region
                intersection = cv2.bitwise_and(region_mask, contour_mask)
                intersection_area = cv2.countNonZero(intersection)
                region_area = (region_y_max - region_y_min) * (region_x_max - region_x_min)
                
                # If significant intersection, consider it an occlusion
                if intersection_area > (region_area * self.occlusion_threshold):
                    # Get bounding box of the contour
                    x, y, w_c, h_c = cv2.boundingRect(contour)
                    
                    # Convert coordinates to original face coordinates
                    occlusion_box = (
                        x_min + x, 
                        y_min + y, 
                        x_min + x + w_c, 
                        y_min + y + h_c
                    )
                    
                    # Calculate confidence score based on multiple factors
                    # Higher for larger intersections, balanced with overall size constraints
                    confidence = min(1.0, (intersection_area / region_area) * (1.0 - abs(0.3 - contour_area_ratio)))
                    
                    # Only add if confidence is high enough (using the threshold from initialization)
                    if confidence > self.confidence_threshold:
                        occlusions.append({
                            'bbox': occlusion_box,
                            'region': region_name,
                            'confidence': confidence
                        })
                    
                    # No need to check more contours for this region
                    break
        
        return occlusions
    
    def draw_occlusions(self, image, occlusions):
        """
        Draw occlusion bounding boxes on the image.
        
        Args:
            image: Original image
            occlusions: List of detected occlusions
            
        Returns:
            Image with drawn occlusions
        """
        annotated_image = image.copy()
        
        for occlusion in occlusions:
            x_min, y_min, x_max, y_max = occlusion['bbox']
            
            # Draw red box around occlusion with slightly thicker lines for visibility
            cv2.rectangle(
                annotated_image, 
                (int(x_min), int(y_min)), 
                (int(x_max), int(y_max)),
                (0, 0, 255),  # Red color for occlusions
                3  # Thicker line for better visibility
            )
            
            # Create a semi-transparent overlay to highlight the occlusion
            overlay = annotated_image.copy()
            cv2.rectangle(
                overlay,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),  # Red color for occlusions
                -1  # Fill the rectangle
            )
            # Apply the overlay with transparency
            alpha = 0.2  # Transparency factor
            cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
            
            # Add a more descriptive label
            label = f"Object covering {occlusion['region']}"
            
            # Add a background for the text to make it more readable
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated_image,
                (int(x_min), int(y_min) - text_size[1] - 10),
                (int(x_min) + text_size[0], int(y_min)),
                (0, 0, 0),
                -1
            )
            
            # Add the label text
            cv2.putText(
                annotated_image,
                label,
                (int(x_min), int(y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text for contrast
                1
            )
        
        return annotated_image
