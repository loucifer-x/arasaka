# Arasaka







# Overview

Arasaka is a real-time computer vision surveillance system that detects and profiles faces from live camera streams. It uses facial recognition to group recurring faces into profiles and can compare them against a configurable watchlist. The system provides a Dear PyGui dashboard for live monitoring, profile management, camera configuration, and watchlist alerts.

![AI Status](https://img.shields.io/badge/AI%20OVERVIEW-DISABLED-red?style=for-the-badge)

> [!WARNING]
> **AI Overview is currently disabled.**
> Face detection, recognition, profiling, and watchlist matching continue to work normally.
> 

## Screenshots

### Live View
![Live View](images/1.png)

### Profiles
![Profiles](images/2.png)

### Watchlist
![Watchlist](images/3.png)

### Camera Manager
![Camera Manager](images/4.png)

### Requirements 

```
curl -sSL -o face_detection_yunet.onnx   "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
pip install git+https://github.com/ageitgey/face_recognition_models
pip install face_recognition
pip install "setuptools<81"
pip install requests
pip install mediapipe
pip install dearpygui
pip install "opencv-python<5"
pip install numpy
pip install aiohttp
pip install Pillow
```
