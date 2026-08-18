# arasaka-
WORK IN PROGRESS






```
                    ┌─────────────────┐
                    │     Camera      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Video Stream   │
                    │ BlenderStream   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Face Detection  │
                    │ face_detector   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Face Encoding   │
                    │ face_recognition│
                    └────────┬────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │   Profile Matching      │
                 │                         │
                 │ Existing face? ── Yes ──┤
                 │        │                │
                 │        No               │
                 └─────────┬───────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      ┌──────────────┐            ┌──────────────┐
      │Update Profile│            │Create Profile│
      │hits / image  │            │new embedding │
      └──────┬───────┘            └──────┬───────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 ┌─────────────────────┐
                 │   Watchlist Match   │
                 └──────────┬──────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                  Match           No Match
                    │               │
                    ▼               ▼
             ┌────────────┐   ┌────────────┐
             │    Ping    │   │   Normal   │
             │   Alert    │   │  Profile   │
             └─────┬──────┘   └────────────┘
                   │
                   ▼
             ┌──────────────┐
             │  AI Overview │
             │   (Optional) │
             │    Ollama    │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │  Dear PyGui  │
             │     GUI      │
             └──────────────┘
```

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
