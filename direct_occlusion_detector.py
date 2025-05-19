import cv2
import numpy as np

class DirectOcclusionDetector:
    """
    A very direct and simple occlusion detector that uses basic image processing
    to detect objects in front of a face.
    """
    def __init__(self):
        """Initialize the direct occlusion detector."""
        # Previous frame for motion detection
        self.prev_frame = None
        
        # Detection threshold - lower = more sensitive
        self.threshold = 40  # For difference between frames
        self.min_area = 200  # Minimum area to be considered an occlusion
        
    def detect_occlusions(self, face_roi, face_bbox):
        """
        Directly detect any objects covering a face.
        
        Args:
            face_roi: Face region of interest
            face_bbox: Face bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            List of detected occlusions
        """
        if face_roi is None or face_roi.size == 0:
            return []
            
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()
            
        # Get face dimensions
        height, width = gray.shape[:2]
        x_min, y_min, x_max, y_max = face_bbox
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Check for motion if we have a previous frame
        occlusions = []
        
        # Method 1: Look for areas of high contrast (potential objects)
        # Calculate the local variance (texture measure)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian = cv2.Laplacian(blur, cv2.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        
        # Threshold to find areas of high contrast
        _, high_contrast = cv2.threshold(laplacian_abs.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
        
        # Find contours in high contrast regions
        contours, _ = cv2.findContours(high_contrast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create potential occlusion regions from larger contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert to global coordinates
                occlusion_box = (
                    x_min + x,
                    y_min + y,
                    x_min + x + w,
                    y_min + y + h
                )
                
                # Determine region name
                if y < height/3:
                    region = "forehead/eyes"
                elif y < 2*height/3:
                    region = "nose"
                else:
                    region = "mouth/chin"
                
                occlusions.append({
                    'bbox': occlusion_box,
                    'region': f"OBJECT over {region}",
                    'confidence': min(1.0, area / 1000)
                })
                
        # Method 2: Frame differencing for motion detection
        if self.prev_frame is not None and self.prev_frame.shape == gray.shape:
            # Calculate difference between frames
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            
            # Threshold to find areas of significant change
            _, motion_mask = cv2.threshold(frame_diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # Dilate to connect nearby motion areas
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(motion_mask, kernel, iterations=2)
            
            # Find motion contours
            motion_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process motion contours
            for contour in motion_contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert to global coordinates
                    occlusion_box = (
                        x_min + x,
                        y_min + y,
                        x_min + x + w,
                        y_min + y + h
                    )
                    
                    # Strong bias toward classifying motion as an occlusion
                    occlusions.append({
                        'bbox': occlusion_box,
                        'region': "MOVING OBJECT",
                        'confidence': 0.9  # High confidence for motion
                    })
        
        # Method 3: If RGB face, directly check for non-skin colors
        if len(face_roi.shape) == 3:
            # Convert to HSV color space
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Define range for skin color
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Invert to get non-skin areas
            non_skin = cv2.bitwise_not(skin_mask)
            
            # Apply morphological operations to clean up
            kernel = np.ones((7, 7), np.uint8)
            non_skin = cv2.morphologyEx(non_skin, cv2.MORPH_CLOSE, kernel)
            non_skin = cv2.morphologyEx(non_skin, cv2.MORPH_OPEN, kernel)
            
            # Find contours in non-skin areas
            skin_contours, _ = cv2.findContours(non_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for large non-skin regions
            for contour in skin_contours:
                area = cv2.contourArea(contour)
                if area > self.min_area * 2:  # Higher threshold for non-skin
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Convert to global coordinates
                    occlusion_box = (
                        x_min + x,
                        y_min + y,
                        x_min + x + w,
                        y_min + y + h
                    )
                    
                    # Determine region name based on y position
                    if y < height/3:
                        region = "forehead/eyes"
                    elif y < 2*height/3:
                        region = "nose"
                    else:
                        region = "mouth/chin"
                    
                    occlusions.append({
                        'bbox': occlusion_box,
                        'region': f"NON-SKIN OBJECT over {region}",
                        'confidence': min(1.0, area / 2000)
                    })
        
        # HACK: If any large object was placed in front of the camera, the entire face
        # region would likely have significant changes. If no specific occlusions were
        # found but the face is present, force an occlusion alert based on total face changes
        if not occlusions and gray is not None and len(gray) > 0:
            # Calculate the standard deviation of the face region - high values
            # often indicate an object is present
            std_dev = np.std(gray)
            
            if std_dev > 60:  # High standard deviation suggests occlusion
                # Mark the entire face as occluded
                occlusions.append({
                    'bbox': face_bbox,
                    'region': "FACE COVERING DETECTED",
                    'confidence': 0.8
                })
                
        # Store the current frame for the next comparison
        self.prev_frame = gray.copy()
        
        # Return the detected occlusions, if any
        return occlusions
                
    def draw_occlusions(self, image, occlusions):
        """Draw detected occlusions on the image."""
        result = image.copy()
        
        for occlusion in occlusions:
            # Get the bounding box
            x_min, y_min, x_max, y_max = occlusion['bbox']
            
            # Draw a thick red rectangle
            cv2.rectangle(
                result,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),  # Red
                3  # Thickness
            )
            
            # Semi-transparent overlay
            overlay = result.copy()
            cv2.rectangle(
                overlay,
                (int(x_min), int(y_min)),
                (int(x_max), int(y_max)),
                (0, 0, 255),
                -1  # Fill
            )
            # Apply overlay with transparency
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, result, 1-alpha, 0, result)
            
            # Add label
            region = occlusion['region']
            conf = occlusion['confidence']
            label = f"{region} ({conf:.2f})"
            
            # Create text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                result,
                (int(x_min), int(y_min) - text_size[1] - 5),
                (int(x_min) + text_size[0], int(y_min)),
                (0, 0, 0),
                -1
            )
            
            # Add text
            cv2.putText(
                result,
                label,
                (int(x_min), int(y_min) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
        return result
