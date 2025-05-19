import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

class EmotionRecognizer:
    """
    Emotion recognition class using a pre-trained model.
    """
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognizer.
        
        Args:
            model_path: Path to the pre-trained emotion recognition model.
                        If None, will use a simple model loaded at runtime.
        """
        self.emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        self.target_size = (48, 48)  # Standard input size for emotion models
        
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
        Create a simple RandomForest model for emotion recognition.
        This is a placeholder and should be replaced with a properly trained model.
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        # Using a fixed feature size for consistency
        self.n_features = 2304  # 48x48 = 2304 (or less for memory efficiency)
        
        # Train the model with a minimal set of random data
        # This is just a placeholder - in a real application, use proper training data
        num_samples = 70  # 10 samples per class for 7 emotions
        X_dummy = np.random.rand(num_samples, self.n_features)  # Fixed feature dimension
        y_dummy = np.repeat(np.arange(len(self.emotion_labels)), 10)  # 10 samples per emotion
        model.fit(X_dummy, y_dummy)
        
        return model
    
    def _save_model(self):
        """Save the model to disk."""
        model_path = os.path.join("models", "emotion_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Emotion model saved to {model_path}")
        
    def _load_model(self, model_path):
        """Load the model from disk."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def preprocess_face(self, face_roi):
        """
        Preprocess the face region for emotion recognition.
        
        Args:
            face_roi: The face region of interest (ROI) from the detector
            
        Returns:
            Preprocessed face image ready for model input
        """
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_roi
        
        # Resize to target size
        face_resized = cv2.resize(face_gray, self.target_size)
        
        # Normalize pixel values
        face_normalized = face_resized / 255.0
        
        # Flatten the image for scikit-learn models
        face_flat = face_normalized.flatten()
        
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
    
    def predict_emotion(self, face_roi):
        """
        Predict emotion from a face ROI.
        
        Args:
            face_roi: The face region of interest (ROI) from the detector
            
        Returns:
            Dictionary containing emotion label and confidence
        """
        # Check if face_roi is empty
        if face_roi is None or face_roi.size == 0:
            return {
                'emotion': 'Unknown',
                'confidence': 0.0,
                'emotion_scores': {}
            }
        
        # Preprocess the face
        face_input = self.preprocess_face(face_roi)
        
        # Reshape for scikit-learn model input (flatten the image)
        face_flat = face_input.reshape(1, -1)
        
        # Get prediction
        emotion_idx = self.model.predict(face_flat)[0]
        emotion_label = self.emotion_labels[emotion_idx]
        
        # Get prediction probabilities
        try:
            prediction_proba = self.model.predict_proba(face_flat)[0]
            confidence = float(prediction_proba[emotion_idx])
            # Create a dictionary of all emotion scores
            emotion_scores = {label: float(score) for label, score in zip(self.emotion_labels, prediction_proba)}
        except:
            # If predict_proba is not available, use a default confidence
            confidence = 0.7  # Default confidence
            emotion_scores = {label: 0.1 for label in self.emotion_labels}
            emotion_scores[emotion_label] = confidence
        
        return {
            'emotion': emotion_label,
            'confidence': confidence,
            'emotion_scores': emotion_scores
        }
