import cv2
import numpy as np
import time

def draw_text(img, text, pos, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.6, thickness=1):
    """
    Draw text with background on an image.
    
    Args:
        img: The image to draw on
        text: The text to draw
        pos: Position (x, y) to draw text at
        bg_color: Background color
        text_color: Text color
        font_scale: Font scale
        thickness: Line thickness
    
    Returns:
        img with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 8), (x + text_w + 8, y), bg_color, -1)
    cv2.putText(img, text, (x + 4, y - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return img

def draw_prediction_results(image, face_info, emotion_result, age_result):
    """
    Draw prediction results (emotion and age) on the image for a face.
    
    Args:
        image: The image to draw on
        face_info: Face detection info including bounding box
        emotion_result: Emotion recognition result
        age_result: Age estimation result
    
    Returns:
        img with prediction results drawn
    """
    x_min, y_min, x_max, y_max = face_info['bbox']
    
    # Get emotion and age results
    emotion = emotion_result['emotion']
    emotion_conf = emotion_result['confidence']
    age_range = age_result['age_range']
    age_conf = age_result['confidence']
    
    # Construct the display text
    emotion_text = f"Emotion: {emotion} ({emotion_conf:.2f})"
    age_text = f"Age: {age_range} ({age_conf:.2f})"
    
    # Draw the emotion text below the face
    image = draw_text(
        image, 
        emotion_text, 
        (x_min, y_max + 20), 
        bg_color=(255, 0, 0), 
        text_color=(255, 255, 255)
    )
    
    # Draw the age text below the emotion text
    image = draw_text(
        image, 
        age_text, 
        (x_min, y_max + 45), 
        bg_color=(0, 128, 0), 
        text_color=(255, 255, 255)
    )
    
    return image

class FPS:
    """
    Class to calculate frames per second.
    """
    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.timestamps = []
        self.fps = 0
    
    def update(self):
        """Update with a new frame timestamp"""
        self.timestamps.append(time.time())
        
        # Keep only the latest avg_frames timestamps
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)
    
    def get_fps(self):
        """Calculate the current FPS based on timestamps"""
        if len(self.timestamps) < 2:
            return 0
        
        # Calculate FPS based on differences between timestamps
        diffs = []
        for i in range(1, len(self.timestamps)):
            diffs.append(self.timestamps[i] - self.timestamps[i-1])
        
        # Calculate average time difference and convert to FPS
        avg_diff = sum(diffs) / len(diffs)
        self.fps = 1 / avg_diff if avg_diff > 0 else 0
        
        return self.fps

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize an image maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width or None
        height: Target height or None
        inter: Interpolation method
    
    Returns:
        Resized image
    """
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)
