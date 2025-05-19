Real-Time Face Detection with Emotion and Age Estimation
Overview
This application performs real-time face analysis with emotion recognition and age estimation. It processes video from your webcam or video files and provides insights about faces in the frame.

Key Features
Multi-Face Detection: Detects all faces in the frame
Emotion Recognition: Identifies 7 emotions (Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral)
Age Estimation: Predicts age ranges from 0-2 years to 70+
Real-time Processing: Processes video streams with minimal delay
Flexible Input: Works with webcam or video files
Demo Mode: Simulated faces for testing without a camera
Temporal Smoothing: Stabilizes predictions to prevent flickering
Installation
Prerequisites
Python 3.7+ (tested on Python 3.13)
OpenCV with video support
Numpy and scikit-learn libraries
Steps
Install Dependencies
CopyInsert
pip install -r requirements.txt
Note: MediaPipe is optional and can enhance face detection accuracy if available. I was unable to test it out because i suck ass in using python

Verify Installation
CopyInsert
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
Running the Application
Standard Mode (Using Webcam)
CopyInsert
python main.py
Demo Mode (No Camera Required)
CopyInsert
python main.py --demo
This runs with simulated faces - perfect for testing without a camera.

Process a Video File
CopyInsert
python main.py --input path/to/video.mp4
Additional Options
CopyInsert
python main.py --show_fps            # Display FPS counter
python main.py --save_video          # Save output to file
python main.py --display_width 1280  # Set display width
python main.py --output_path output.mp4  # Custom output path
Controls
Press q to quit the application
Close the video window to end processing
Troubleshooting
No Camera Detected
If the application can't access your webcam, it will automatically:

Try an alternate camera (if multiple are connected)
Fall back to test pattern or demo mode
Manually run demo mode if you experience camera issues:

CopyInsert
python main.py --demo
Performance Issues
If the application runs slowly:

Lower the display resolution: --display_width 640
Make sure your system meets the minimum requirements
Project Structure
main.py: Application entry point
face_detector.py: Face detection module (OpenCV/MediaPipe)
emotion_recognizer.py: Emotion classification
age_estimator.py: Age range estimation
prediction_smoother.py: Stabilizes predictions over time
utils.py: Helper functions and visualization tools
models/: Directory for model storage
Please note that camera quality may disrupt prediction,
You need to install python for this to work!
Download here:
https://www.python.org/downloads/
---end----
