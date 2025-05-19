import numpy as np
from collections import deque

class PredictionSmoother:
    """
    Class to smooth predictions over time to reduce fluctuations.
    """
    def __init__(self, history_size=10):
        """
        Initialize the prediction smoother.
        
        Args:
            history_size: Number of frames to consider for smoothing
        """
        self.history_size = history_size
        self.emotion_history = {}
        self.age_history = {}
        self.face_trackers = {}
        self.next_id = 0
    
    def _assign_face_id(self, face_bbox, existing_faces):
        """
        Assign an ID to a face by matching it with existing faces.
        Simple IOU-based tracking.
        
        Args:
            face_bbox: Current face bounding box (x_min, y_min, x_max, y_max)
            existing_faces: Dictionary of existing face IDs and their bounding boxes
            
        Returns:
            Best matching face ID, or a new ID if no good match is found
        """
        best_match_id = None
        best_iou = 0.3  # Minimum IOU threshold for a match
        
        # Calculate current face box area
        x_min, y_min, x_max, y_max = face_bbox
        area = (x_max - x_min) * (y_max - y_min)
        
        # Check for matches with existing faces
        for face_id, prev_bbox in existing_faces.items():
            prev_x_min, prev_y_min, prev_x_max, prev_y_max = prev_bbox
            
            # Calculate intersection
            int_x_min = max(x_min, prev_x_min)
            int_y_min = max(y_min, prev_y_min)
            int_x_max = min(x_max, prev_x_max)
            int_y_max = min(y_max, prev_y_max)
            
            # Skip if no intersection
            if int_x_max <= int_x_min or int_y_max <= int_y_min:
                continue
                
            int_area = (int_x_max - int_x_min) * (int_y_max - int_y_min)
            prev_area = (prev_x_max - prev_x_min) * (prev_y_max - prev_y_min)
            union_area = area + prev_area - int_area
            
            # Calculate IOU
            iou = int_area / union_area if union_area > 0 else 0
            
            # Update best match if this is better
            if iou > best_iou:
                best_iou = iou
                best_match_id = face_id
        
        # If no good match found, assign a new ID
        if best_match_id is None:
            best_match_id = self.next_id
            self.next_id += 1
            
        return best_match_id
    
    def update_face_tracking(self, faces):
        """
        Update face tracking across frames.
        
        Args:
            faces: List of detected faces with bounding boxes
            
        Returns:
            Updated faces list with tracking IDs
        """
        # Create a dictionary of current faces
        current_faces = {}
        
        # First pass: try to match existing IDs
        for face in faces:
            face_bbox = face['bbox']
            face_id = self._assign_face_id(face_bbox, self.face_trackers)
            
            # Store the tracking ID in the face info
            face['tracking_id'] = face_id
            
            # Update the trackers
            current_faces[face_id] = face_bbox
        
        # Replace the old trackers with the current ones
        self.face_trackers = current_faces
        
        return faces
    
    def smooth_emotion_prediction(self, face_id, emotion_result):
        """
        Smooth emotion predictions over time.
        
        Args:
            face_id: Face tracking ID
            emotion_result: Current emotion prediction result
            
        Returns:
            Smoothed emotion prediction
        """
        # Initialize history for this face if it doesn't exist
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = {
                'labels': deque(maxlen=self.history_size),
                'scores': {}
            }
        
        history = self.emotion_history[face_id]
        
        # Add current prediction to history
        current_emotion = emotion_result['emotion']
        history['labels'].append(current_emotion)
        
        # Update scores history
        for emotion, score in emotion_result['emotion_scores'].items():
            if emotion not in history['scores']:
                history['scores'][emotion] = deque([score], maxlen=self.history_size)
            else:
                history['scores'][emotion].append(score)
        
        # Count occurrences of each emotion in history
        emotion_counts = {}
        for emotion in history['labels']:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
        
        # Find the most frequent emotion
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        smoothed_emotion = sorted_emotions[0][0]
        
        # Calculate average scores
        smoothed_scores = {}
        for emotion, scores in history['scores'].items():
            smoothed_scores[emotion] = sum(scores) / len(scores)
        
        # Return smoothed result
        return {
            'emotion': smoothed_emotion,
            'confidence': smoothed_scores.get(smoothed_emotion, 0.0),
            'emotion_scores': smoothed_scores
        }
    
    def smooth_age_prediction(self, face_id, age_result):
        """
        Smooth age predictions over time.
        
        Args:
            face_id: Face tracking ID
            age_result: Current age prediction result
            
        Returns:
            Smoothed age prediction
        """
        # Initialize history for this face if it doesn't exist
        if face_id not in self.age_history:
            self.age_history[face_id] = {
                'ranges': deque(maxlen=self.history_size),
                'scores': {}
            }
        
        history = self.age_history[face_id]
        
        # Add current prediction to history
        current_age_range = age_result['age_range']
        history['ranges'].append(current_age_range)
        
        # Update scores history
        for age_range, score in age_result['age_range_scores'].items():
            if age_range not in history['scores']:
                history['scores'][age_range] = deque([score], maxlen=self.history_size)
            else:
                history['scores'][age_range].append(score)
        
        # Count occurrences of each age range in history
        range_counts = {}
        for age_range in history['ranges']:
            if age_range in range_counts:
                range_counts[age_range] += 1
            else:
                range_counts[age_range] = 1
        
        # Find the most frequent age range
        sorted_ranges = sorted(range_counts.items(), key=lambda x: x[1], reverse=True)
        smoothed_age_range = sorted_ranges[0][0]
        
        # Calculate average scores
        smoothed_scores = {}
        for age_range, scores in history['scores'].items():
            smoothed_scores[age_range] = sum(scores) / len(scores)
        
        # Return smoothed result
        return {
            'age_range': smoothed_age_range,
            'confidence': smoothed_scores.get(smoothed_age_range, 0.0),
            'age_range_scores': smoothed_scores
        }
    
    def clean_history(self, active_face_ids):
        """
        Clean up history for faces that are no longer detected.
        
        Args:
            active_face_ids: List of face IDs that are currently active
        """
        # Convert to set for faster lookup
        active_ids = set(active_face_ids)
        
        # Clean emotion history
        emotion_keys = list(self.emotion_history.keys())
        for face_id in emotion_keys:
            if face_id not in active_ids:
                del self.emotion_history[face_id]
        
        # Clean age history
        age_keys = list(self.age_history.keys())
        for face_id in age_keys:
            if face_id not in active_ids:
                del self.age_history[face_id]
