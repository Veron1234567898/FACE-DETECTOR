# Real-Time Face Detection with Emotion and Age Estimation

This project implements a real-time face analysis system that can:
- Detect multiple faces in a video stream
- Estimate the emotion of each detected face
- Estimate the age of each detected person
- Annotate faces with bounding boxes and overlay predictions

## Features

- **Face Detection**: Uses MediaPipe Face Detection for robust and fast detection of multiple faces
- **Emotion Recognition**: Classifies emotions into 7 categories (Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral)
- **Age Estimation**: Predicts age ranges for detected faces
- **Real-time Processing**: Works with webcam feed or video files
- **Visualization**: Displays bounding boxes with emotion and age information

## Requirements

```
pip install -r requirements.txt
```

## Usage

To run with webcam:
```
python main.py
```

To run with a video file:
```
python main.py --input path/to/video.mp4
```

Additional options:
```
python main.py --help
```

## Project Structure

- `main.py`: Entry point of the application
- `face_detector.py`: Module for face detection using MediaPipe
- `emotion_recognizer.py`: Module for emotion recognition
- `age_estimator.py`: Module for age estimation
- `utils.py`: Utility functions for visualization and processing
- `models/`: Directory containing pre-trained models

## License

MIT
