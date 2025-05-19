import cv2
import argparse
import os
import time
import numpy as np
import random
from face_detector import FaceDetector
from emotion_recognizer import EmotionRecognizer
from age_estimator import AgeEstimator
from prediction_smoother import PredictionSmoother
from utils import draw_prediction_results, FPS, resize_with_aspect_ratio

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time Face Detection with Emotion and Age Estimation')
    parser.add_argument('--input', type=str, default='0',
                        help='Path to video file or webcam index (default: 0)')
    parser.add_argument('--display_width', type=int, default=960,
                        help='Width of the display window (default: 960)')
    parser.add_argument('--show_fps', action='store_true',
                        help='Display FPS on output frame')
    parser.add_argument('--save_video', action='store_true',
                        help='Save output video to file')
    parser.add_argument('--output_path', type=str, default='output.mp4',
                        help='Path to save output video (default: output.mp4)')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with animated faces')
    parser.add_argument('--processing_width', type=int, default=320,
                        help='Width to resize frames for processing (smaller = faster)')
    parser.add_argument('--process_every', type=int, default=2,
                        help='Process every Nth frame for detection (higher = faster)')
    parser.add_argument('--occlusion_detection', action='store_true', help='Enable detection of objects covering the face')

    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check if demo mode is requested
    if args.demo:
        print("Running in demo mode with animated faces.")
        use_demo_mode = True
        use_test_pattern = False
    else:
        use_demo_mode = False
        use_test_pattern = False
    
    # Initialize video capture
    # Try to convert input to integer for webcam index
    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input  # Use as path to video file
    
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        print("Trying alternate video source...")
        # Try another camera index
        if isinstance(input_source, int):
            alt_source = 1 if input_source == 0 else 0
            cap = cv2.VideoCapture(alt_source)
            
        # If still can't open a camera, use a static image or test pattern
        if not cap.isOpened():
            print("Could not open any camera. Using test pattern instead.")
            # Create a test image with a face-like pattern
            use_test_pattern = True
            frame_width, frame_height = 640, 480
            fps = 30
            # Release the failed capture
            if 'cap' in locals() and cap is not None:
                cap.release()
        else:
            use_test_pattern = False
    else:
        use_test_pattern = False
    
    # Get video properties if not using test pattern
    if not use_test_pattern:
        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        except:
            # Default values if camera properties can't be read
            frame_width, frame_height = 640, 480
            fps = 30
    
    # Create output video writer if needed
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            args.output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize face detector and recognizers with optimized parameters
    face_detector = FaceDetector(min_detection_confidence=0.5)
    face_detector.detect_occlusions = args.occlusion_detection
    emotion_recognizer = EmotionRecognizer()
    age_estimator = AgeEstimator()
    
    # Processing parameters for performance optimization
    process_width = args.processing_width
    process_every = args.process_every
    frame_count = 0
    last_frame_time = time.time()
    processing_frame = None  # Store the frame being processed
    
    # FPS calculator
    fps_counter = FPS()
    
    # Initialize prediction smoother
    prediction_smoother = PredictionSmoother(history_size=15)
    
    # Initialize variables for demo mode
    if use_demo_mode:
        # Create some demo faces at fixed positions
        demo_faces = [
            {'x': frame_width // 4, 'y': frame_height // 4, 'size': 80},
            {'x': 3 * frame_width // 4, 'y': frame_height // 4, 'size': 70},
            {'x': frame_width // 2, 'y': 3 * frame_height // 4, 'size': 90}
        ]
        # Emotions and ages to cycle through
        demo_emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Neutral']
        demo_ages = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        frame_count = 0
    
    # Set up window
    window_name = "Real-time Face Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, frame_width, frame_height)
    
    # Print status information
    print(f"Starting video processing with:")
    print(f" - Processing every {process_every} frame(s)")
    print(f" - Processing width: {process_width} pixels")
    print(f" - Occlusion detection: {'Enabled' if args.occlusion_detection else 'Disabled'}") 
    print("Press 'q' to quit.")
    
    while True:
        current_time = time.time()
        frame_time_delta = current_time - last_frame_time
        last_frame_time = current_time
        
        if use_demo_mode:
            # Create a blank frame with gray background
            frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 64
            
            # Update frame counter
            frame_count += 1
            
            # Add frame info
            cv2.putText(frame, f"Demo Mode - Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Initialize detected faces list
            detected_faces = []
            
            # Process each demo face
            for i, face in enumerate(demo_faces):
                # Move faces slightly to simulate motion
                if frame_count % 10 == 0:  # Move every 10 frames
                    face['x'] += random.randint(-5, 5)
                    face['y'] += random.randint(-5, 5)
                    
                    # Keep faces within frame
                    face['x'] = max(face['size'], min(frame_width - face['size'], face['x']))
                    face['y'] = max(face['size'], min(frame_height - face['size'], face['y']))
                
                # Draw face
                x, y, size = face['x'], face['y'], face['size']
                x_min = max(0, x - size)
                y_min = max(0, y - size)
                x_max = min(frame_width, x + size)
                y_max = min(frame_height, y + size)
                
                # Draw simple face visualization
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (200, 200, 0), 2)
                
                # Extract region
                face_roi = frame[y_min:y_max, x_min:x_max].copy()
                if face_roi.size > 0:
                    # Create demo detection with cycling emotions and ages
                    emotion = demo_emotions[(i + frame_count // 30) % len(demo_emotions)]
                    age = demo_ages[(i + frame_count // 50) % len(demo_ages)]
                    
                    # Create face info dictionary as if from detector
                    face_info = {
                        'bbox': (x_min, y_min, x_max, y_max),
                        'confidence': 0.95,
                        'landmarks': {
                            'left_eye': (x_min + size // 3, y_min + size // 3),
                            'right_eye': (x_max - size // 3, y_min + size // 3),
                            'nose_tip': (x, y),
                            'mouth_center': (x, y_max - size // 3)
                        },
                        'face_roi': face_roi,
                        'emotion': emotion,
                        'age': age
                    }
                    
                    # Add to detected faces
                    detected_faces.append(face_info)
            
            ret = True  # Always valid in demo mode
        elif use_test_pattern:
            # Create a test image with a face-like pattern
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Draw a simple face pattern
            cv2.rectangle(frame, (frame_width//2 - 100, frame_height//2 - 130), (frame_width//2 + 100, frame_height//2 + 130), (200, 200, 0), 2)
            cv2.circle(frame, (frame_width//2 - 40, frame_height//2 - 30), 20, (255, 255, 255), -1)  # Left eye
            cv2.circle(frame, (frame_width//2 + 40, frame_height//2 - 30), 20, (255, 255, 255), -1)  # Right eye
            cv2.ellipse(frame, (frame_width//2, frame_height//2 + 20), (60, 30), 0, 0, 180, (0, 0, 255), -1)  # Mouth
            
            # Add text to indicate it's a test pattern
            cv2.putText(frame, "Test Pattern - No Camera", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret = True
        else:
            # Read frame from camera or video file
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break
        
        # Start timing for this frame
        start_time = time.time()
        
        # Only process every Nth frame to improve framerate (skip frames)
        # But always draw using the latest detection results
        if not use_demo_mode and (frame_count % process_every == 0):
            # Get smaller frame for processing to improve speed
            if frame.shape[1] > process_width:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                process_height = int(process_width * aspect_ratio)
                processing_frame = cv2.resize(frame, (process_width, process_height))
            else:
                processing_frame = frame.copy()
                
            # Detect faces on the smaller frame
            detected_faces = face_detector.detect_faces(processing_frame)
            
            # Scale bounding boxes back to original frame size if needed
            if frame.shape[1] > process_width:
                scale_x = frame.shape[1] / process_width
                scale_y = frame.shape[0] / process_height
                
                for face in detected_faces:
                    x_min, y_min, x_max, y_max = face['bbox']
                    x_min = int(x_min * scale_x)
                    y_min = int(y_min * scale_y)
                    x_max = int(x_max * scale_x)
                    y_max = int(y_max * scale_y)
                    
                    # Update bbox
                    face['bbox'] = (x_min, y_min, x_max, y_max)
                    
                    # Update landmarks
                    for key in face['landmarks']:
                        x, y = face['landmarks'][key]
                        face['landmarks'][key] = (int(x * scale_x), int(y * scale_y))
                    
                    # Extract face ROI from original frame
                    face['face_roi'] = frame[y_min:y_max, x_min:x_max] if 0 <= y_min < y_max < frame.shape[0] and 0 <= x_min < x_max < frame.shape[1] else None
            
        # Increment frame counter
        frame_count += 1
        
        # Apply face tracking to maintain consistent IDs
        tracked_faces = prediction_smoother.update_face_tracking(detected_faces)
        
        # Process each detected face
        active_face_ids = []
        for face_info in tracked_faces:
            # Get tracking ID
            face_id = face_info['tracking_id']
            active_face_ids.append(face_id)
            
            # Get face region
            face_roi = face_info.get('face_roi')
            
            if face_roi is not None and face_roi.size > 0:
                # Get raw predictions
                raw_emotion_result = emotion_recognizer.predict_emotion(face_roi)
                raw_age_result = age_estimator.predict_age(face_roi)
                
                # Apply temporal smoothing
                emotion_result = prediction_smoother.smooth_emotion_prediction(face_id, raw_emotion_result)
                age_result = prediction_smoother.smooth_age_prediction(face_id, raw_age_result)
                
                # Check if there are any occlusions detected and print debug info
                if 'occlusions' in face_info and face_info['occlusions']:
                    print(f"Main.py: Found {len(face_info['occlusions'])} occlusions to display")
                    
                    # Draw a VERY obvious warning about occlusions
                    # Add a red banner across the top of the screen
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 255), -1)
                    
                    # Add white text on the red background
                    cv2.putText(
                        frame,
                        f"OBJECT COVERING FACE DETECTED!",
                        (frame.shape[1]//2 - 240, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),  # White color
                        2
                    )
                    
                    # Draw each occlusion directly on the frame
                    for occlusion in face_info['occlusions']:
                        # Get bounding box
                        x_min, y_min, x_max, y_max = occlusion['bbox']
                        
                        # Draw a thick bright red box around the occlusion
                        cv2.rectangle(
                            frame,
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (0, 0, 255),  # Red color
                            4  # Very thick
                        )
                        
                        # Add a semi-transparent red overlay
                        overlay = frame.copy()
                        cv2.rectangle(
                            overlay,
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (0, 0, 255),  # Red
                            -1  # Fill
                        )
                        # Apply with transparency
                        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                        
                        # Add text label above the box
                        cv2.putText(
                            frame,
                            f"OBJECT",
                            (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),  # Red
                            2  # Thick
                        )
                else:
                    print("Main.py: No occlusions found in face_info")
                
                # Draw bounding box, landmarks, and occlusions if detected
                frame = face_detector.draw_detections(frame, [face_info], show_confidence=False)
                
                # Draw prediction results
                frame = draw_prediction_results(frame, face_info, emotion_result, age_result)
        
        # Clean up history for faces that are no longer detected
        prediction_smoother.clean_history(active_face_ids)
        
        # Calculate FPS
        fps_counter.update()
        if args.show_fps:
            fps_text = f"FPS: {fps_counter.get_fps():.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
        
        # Resize frame for display
        display_frame = resize_with_aspect_ratio(frame, width=args.display_width)
        
        # Display the frame
        cv2.imshow('Face Analysis', display_frame)
        
        # Write frame to output video if saving
        if args.save_video:
            out.write(frame)
        
        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
