# Arasaka

**Here is what the technology enables on a 10-year-old laptop**

This isn't futuristic technology reserved for governments or huge technology companies. Many of the building blocks required for sophisticated camera-based observation are already accessible to individuals with relatively modest hardware and an internet connection. This project demonstrates how ANY ordinary camera can be turned into an invasive observation tool. Adding a local LLM to the pipeline can exponentially increase the depth and versatility of the data extracted from each frame.

The purpose isn't to demonize these technologies, but to make their capabilities some what tangible. By demonstrating what can be observed, recorded the project explores the increasingly important boundary between legitimate security applications and individual privacy.


# Overview

Arasaka is a real-time computer vision surveillance system that detects and profiles faces from live camera streams. It uses facial recognition to group recurring faces into profiles and can compare them against a configurable watchlist. The system provides a Dear PyGui dashboard for live monitoring, profile management, camera configuration, and watchlist alerts.



![AI Status](https://img.shields.io/badge/AI%20OVERVIEW-DISABLED-red?style=for-the-badge)

> [!WARNING]
> **AI Overview is currently disabled.**
> Face detection, recognition, profiling, and watchlist matching continue to work normally.
> 

## UI Preview

### Live CCTV
Monitor the normal live CCTV stream with real-time face detection.

![Live CCTV](images/4.png)

### Watchlist Enrollment
Set up potential watchlist hits by capturing faces from the live stream or ingesting existing photos.

![Watchlist Enrollment](images/3.png)

### Watchlist Match Hits
View successful matches against configured watchlist entries.

![Watchlist Match Hits](images/2.png)

### Camera Configuration
Add and configure camera streams for the system.

![Camera Configuration](images/1.png)

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
