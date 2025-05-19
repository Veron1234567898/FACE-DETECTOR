import cv2
import numpy as np
import random
import time

class DemoFaceGenerator:
    """
    Generate demo faces for testing the facial analysis system without a camera.
    """
    def __init__(self, width=640, height=480):
        """
        Initialize the demo face generator.
        
        Args:
            width: Frame width
            height: Frame height
        """
        self.width = width
        self.height = height
        self.faces = []
        self.frame_count = 0
        self.max_faces = 3
        self.background_color = (40, 40, 40)  # Dark background
        
        # Create initial faces
        self._create_random_faces(2)  # Start with 2 faces
    
    def _create_face(self, x, y, size, vx, vy):
        """Create a face with random appearance."""
        # Random face color (human skin tone range)
        r = random.randint(180, 240)
        g = random.randint(140, 200)
        b = random.randint(100, 160)
        face_color = (b, g, r)  # OpenCV uses BGR
        
        # Random eye color
        eye_r = random.randint(0, 100)
        eye_g = random.randint(0, 100)
        eye_b = random.randint(0, 255)
        eye_color = (eye_b, eye_g, eye_r)
        
        # Random mouth color
        mouth_r = random.randint(150, 255)
        mouth_g = random.randint(0, 100)
        mouth_b = random.randint(0, 100)
        mouth_color = (mouth_b, mouth_g, mouth_r)
        
        # Random "emotion" - affects mouth curvature
        emotion = random.choice(['happy', 'sad', 'neutral', 'surprised'])
        # This will control the drawing parameters
        
        # Random "age" - affects face size and details
        age = random.randint(0, 80)
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'vx': vx,  # Velocity x
            'vy': vy,  # Velocity y
            'face_color': face_color,
            'eye_color': eye_color,
            'mouth_color': mouth_color,
            'emotion': emotion,
            'age': age,
            'blink_timer': 0,
            'is_blinking': False,
            'expression_change_timer': random.randint(20, 100)
        }
    
    def _create_random_faces(self, count):
        """Create random faces."""
        for _ in range(count):
            # Random position
            x = random.randint(100, self.width - 100)
            y = random.randint(100, self.height - 100)
            
            # Random size (50-100 pixels radius)
            size = random.randint(50, 100)
            
            # Random velocity (-2 to 2 pixels per frame)
            vx = random.uniform(-2, 2)
            vy = random.uniform(-2, 2)
            
            # Create and add the face
            self.faces.append(self._create_face(x, y, size, vx, vy))
    
    def _update_faces(self):
        """Update face positions and states."""
        # Randomly add or remove faces
        if random.random() < 0.01 and len(self.faces) < self.max_faces:
            self._create_random_faces(1)
        elif random.random() < 0.01 and len(self.faces) > 1:
            self.faces.pop(random.randint(0, len(self.faces) - 1))
        
        # Update existing faces
        for face in self.faces:
            # Move the face
            face['x'] += face['vx']
            face['y'] += face['vy']
            
            # Bounce off edges
            if face['x'] - face['size'] < 0 or face['x'] + face['size'] > self.width:
                face['vx'] = -face['vx']
                face['x'] += face['vx']  # Adjust to avoid sticking to edge
            
            if face['y'] - face['size'] < 0 or face['y'] + face['size'] > self.height:
                face['vy'] = -face['vy']
                face['y'] += face['vy']  # Adjust to avoid sticking to edge
            
            # Handle blinking
            if not face['is_blinking'] and random.random() < 0.01:
                face['is_blinking'] = True
                face['blink_timer'] = 5  # Blink for 5 frames
            
            if face['is_blinking']:
                face['blink_timer'] -= 1
                if face['blink_timer'] <= 0:
                    face['is_blinking'] = False
                    face['blink_timer'] = 0
            
            # Change expression occasionally
            face['expression_change_timer'] -= 1
            if face['expression_change_timer'] <= 0:
                face['emotion'] = random.choice(['happy', 'sad', 'neutral', 'surprised'])
                face['expression_change_timer'] = random.randint(50, 200)
                
                # Slightly change velocity
                face['vx'] += random.uniform(-0.5, 0.5)
                face['vy'] += random.uniform(-0.5, 0.5)
                
                # Keep velocity in reasonable range
                face['vx'] = max(-3, min(3, face['vx']))
                face['vy'] = max(-3, min(3, face['vy']))
    
    def _draw_face(self, frame, face):
        """Draw a face on the frame."""
        x, y, size = int(face['x']), int(face['y']), int(face['size'])
        
        # Draw face - using circle instead of ellipse for better compatibility
        cv2.circle(frame, (x, y), size, face['face_color'], -1)
        
        # Calculate eye position based on face size
        eye_size = max(5, size // 5)
        eye_y = y - int(size * 0.2)
        left_eye_x = x - int(size * 0.4)
        right_eye_x = x + int(size * 0.4)
        
        # Draw eyes (or closed eyes if blinking)
        if face['is_blinking']:
            # Draw closed eyes (lines)
            cv2.line(frame, 
                    (left_eye_x - eye_size, eye_y),
                    (left_eye_x + eye_size, eye_y),
                    (0, 0, 0), 2)
            cv2.line(frame, 
                    (right_eye_x - eye_size, eye_y),
                    (right_eye_x + eye_size, eye_y),
                    (0, 0, 0), 2)
        else:
            # Draw open eyes
            cv2.circle(frame, (left_eye_x, eye_y), eye_size, (255, 255, 255), -1)  # White part
            cv2.circle(frame, (right_eye_x, eye_y), eye_size, (255, 255, 255), -1)  # White part
            
            # Draw pupils
            pupil_size = max(2, eye_size // 2)
            cv2.circle(frame, (left_eye_x, eye_y), pupil_size, face['eye_color'], -1)
            cv2.circle(frame, (right_eye_x, eye_y), pupil_size, face['eye_color'], -1)
        
        # Draw eyebrows
        eyebrow_y = eye_y - int(size * 0.15)
        cv2.line(frame, 
                (left_eye_x - eye_size, eyebrow_y),
                (left_eye_x + eye_size, eyebrow_y),
                (0, 0, 0), 2)
        cv2.line(frame, 
                (right_eye_x - eye_size, eyebrow_y),
                (right_eye_x + eye_size, eyebrow_y),
                (0, 0, 0), 2)
        
        # Draw mouth based on emotion
        mouth_width = int(size * 0.6)
        mouth_height = int(size * 0.2)
        mouth_y = y + int(size * 0.4)
        
        if face['emotion'] == 'happy':
            # Smile - draw a curved line instead of ellipse
            # Create points for a curved smile
            smile_points = []
            for i in range(-mouth_width//2, mouth_width//2 + 1, 5):
                # Parabolic curve for smile
                curve_y = mouth_y + int((i*i) / (mouth_width/2)) // 4
                smile_points.append((x + i, curve_y))
            
            # Draw the smile as a polyline
            if len(smile_points) > 1:
                cv2.polylines(frame, [np.array(smile_points)], False, face['mouth_color'], 3)
                
        elif face['emotion'] == 'sad':
            # Frown - draw an inverted curved line
            frown_points = []
            for i in range(-mouth_width//2, mouth_width//2 + 1, 5):
                # Inverted parabolic curve for frown
                curve_y = mouth_y - int((i*i) / (mouth_width/2)) // 4 + mouth_height
                frown_points.append((x + i, curve_y))
            
            # Draw the frown as a polyline
            if len(frown_points) > 1:
                cv2.polylines(frame, [np.array(frown_points)], False, face['mouth_color'], 3)
                
        elif face['emotion'] == 'surprised':
            # O shape - use a circle
            cv2.circle(frame, (x, mouth_y), mouth_height, face['mouth_color'], -1)
        else:  # neutral
            # Straight line
            cv2.line(frame, 
                   (x - mouth_width//2, mouth_y),
                   (x + mouth_width//2, mouth_y),
                   face['mouth_color'], 3)
        
        # Draw nose
        nose_size = max(3, size // 10)
        nose_y = y + int(size * 0.1)
        cv2.circle(frame, (x, nose_y), nose_size, (80, 80, 80), -1)
        
        # Return the face bounding box and data for detection
        x_min = x - size
        y_min = y - int(size * 1.3)
        x_max = x + size
        y_max = y + int(size * 1.3)
        
        return {
            'bbox': (x_min, y_min, x_max, y_max),
            'confidence': 0.98,
            'landmarks': {
                'left_eye': (left_eye_x, eye_y),
                'right_eye': (right_eye_x, eye_y),
                'nose_tip': (x, nose_y),
                'mouth_center': (x, mouth_y)
            },
            'face_roi': frame[y_min:y_max, x_min:x_max] if (
                0 <= y_min < y_max < frame.shape[0] and 
                0 <= x_min < x_max < frame.shape[1]
            ) else None,
            'emotion': face['emotion'],
            'age': face['age']
        }
    
    def generate_frame(self):
        """Generate a frame with animated faces."""
        # Create blank frame
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * self.background_color
        
        # Update face positions and states
        self._update_faces()
        
        # Draw faces and get detection information
        detections = []
        for face in self.faces:
            detection = self._draw_face(frame, face)
            if detection['face_roi'] is not None:
                detections.append(detection)
        
        # Add frame counter and info
        self.frame_count += 1
        cv2.putText(frame, f"Demo Mode - Frame: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {timestamp}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10, self.height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return frame, detections
