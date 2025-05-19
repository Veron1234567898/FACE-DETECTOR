import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

class AgeEstimator:
    """
    Age estimation class using a pre-trained model.
    """
    def __init__(self, model_path=None):
        """
        Initialize the age estimator.
        
        Args:
            model_path: Path to the pre-trained age estimation model.
                        If None, will use a simple model loaded at runtime.
        """
        # Define age ranges
        self.age_ranges = [
            '0-2', '3-9', '10-19', '20-29', '30-39', 
            '40-49', '50-59', '60-69', '70+'
        ]
        
        self.target_size = (224, 224)  # Standard input size for many age models
        
        # Create the models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Load or create the model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            # If no model is provided, create a simple model
            self.model = self._create_simple_model()
            self._save_model()
    
    def _create_simple_model(self):
        """
        Create a simple RandomForest model for age estimation.
        This is a placeholder and should be replaced with a properly trained model.
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        # Determine the number of features based on our preprocessing
        # Using a lower-dimensional representation to save memory and computation
        self.n_features = 10000  # Fixed feature size used for both training and prediction
        
        # Train the model with a minimal set of random data
        # This is just a placeholder - in a real application, use proper training data
        num_samples = 90  # 10 samples per class for 9 age ranges
        X_dummy = np.random.rand(num_samples, self.n_features)  # Fixed feature size
        y_dummy = np.repeat(np.arange(len(self.age_ranges)), 10)  # 10 samples per age range
        model.fit(X_dummy, y_dummy)
        
        return model
    
    def _save_model(self):
        """Save the model to disk."""
        model_path = os.path.join("models", "age_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Age model saved to {model_path}")
        
    def _load_model(self, model_path):
        """Load the model from disk."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def preprocess_face(self, face_roi):
        """
        Preprocess the face region for age estimation.
        
        Args:
            face_roi: The face region of interest (ROI) from the detector
            
        Returns:
            Preprocessed face image ready for model input
        """
        # Check if face_roi is valid
        if face_roi is None or face_roi.size == 0:
            return None
        
        # Resize to a smaller size to reduce dimensionality
        smaller_size = (64, 64)  # Much smaller than original target_size
        face_resized = cv2.resize(face_roi, smaller_size)
        
        # Convert to RGB if it's BGR
        if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
            # Check if the image is BGR (OpenCV format)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_resized
        
        # Basic normalization
        face_normalized = face_rgb / 255.0
        
        # Flatten for scikit-learn models
        face_flat = face_normalized.reshape(-1)  # Flatten the image
        
        # Ensure consistent feature size with what the model was trained on
        if len(face_flat) > self.n_features:
            # If too many features, sample them to reduce dimensionality
            indices = np.linspace(0, len(face_flat)-1, self.n_features, dtype=int)
            face_flat = face_flat[indices]
        elif len(face_flat) < self.n_features:
            # If too few features, pad with zeros
            padded = np.zeros(self.n_features)
            padded[:len(face_flat)] = face_flat
            face_flat = padded
        
        return face_flat
    
    def predict_age(self, face_roi):
        """
        Predict age from a face ROI.
        
        Args:
            face_roi: The face region of interest (ROI) from the detector
            
        Returns:
            Dictionary containing age range and confidence
        """
        # Check if face_roi is empty
        if face_roi is None or face_roi.size == 0:
            return {
                'age_range': 'Unknown',
                'confidence': 0.0,
                'age_range_scores': {}
            }
        
        # Preprocess the face
        face_input = self.preprocess_face(face_roi)
        
        if face_input is None:
            return {
                'age_range': 'Unknown',
                'confidence': 0.0,
                'age_range_scores': {}
            }
        
        # Reshape for scikit-learn model input
        face_flat = face_input.reshape(1, -1)
        
        # Get prediction
        age_range_idx = self.model.predict(face_flat)[0]
        age_range = self.age_ranges[age_range_idx]
        
        # Get prediction probabilities
        try:
            prediction_proba = self.model.predict_proba(face_flat)[0]
            confidence = float(prediction_proba[age_range_idx])
            # Create a dictionary of all age range scores
            age_range_scores = {label: float(score) for label, score in zip(self.age_ranges, prediction_proba)}
        except:
            # If predict_proba is not available, use a default confidence
            confidence = 0.7  # Default confidence
            age_range_scores = {label: 0.1 for label in self.age_ranges}
            age_range_scores[age_range] = confidence
        
        return {
            'age_range': age_range,
            'confidence': confidence,
            'age_range_scores': age_range_scores
        }
